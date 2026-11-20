"""Frozen 100-image inference, using the original payload and attack registry."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_model(root, name, device):
    module = load_module(Path(__file__).resolve().parents[1] / name / 'adapter.py', f'run_{name}')
    if name == 'redmark':
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        model = module.ReDMark()
        return model, None
    if name == 'sepmark':
        model = module.SepMark()
        model.encoder.to(device)
        model.decoder.to(device)

        @torch.inference_mode()
        def embed(rgb, bits):
            small = module.small(rgb).to(device)
            message = torch.from_numpy(bits.astype(np.float32))[None].to(device) * .2 - .1
            marked = model.encoder(small, message)
            # Original adapter interpolates and quantizes on CPU.
            residual = module.F.interpolate(((marked-small)/2).cpu(), rgb.shape[:2],
                                            mode='bilinear', align_corners=False, antialias=True)
            return np.rint(np.clip(rgb.astype(np.float32)/255 + residual[0].permute(1, 2, 0).numpy(), 0, 1)*255).astype(np.uint8)

        @torch.inference_mode()
        def extract(rgb):
            return model.decoder(module.small(rgb).to(device))[0].cpu().numpy(), 0.

        model.embed = embed
        model.extract = extract
        return model, None
    model = module.RoSteALS()
    for name in ('ae', 'control', 'decoder'):
        getattr(model, name).to(device)

    @torch.inference_mode()
    def embed(rgb, packet):
        small = (module.resize_rgb(rgb, 256)*2-1).to(device)
        latent = model.ae.encode(small)
        message = torch.tensor(packet, dtype=torch.float32, device=device)[None]
        residual, _ = model.control(None, message)
        marked = model.ae.decode(latent+residual).clamp(-1, 1)
        delta = module.F.interpolate(((marked-small)/2).cpu(), rgb.shape[:2],
                                     mode='bilinear', align_corners=False, antialias=True)
        return np.rint(np.clip(rgb.astype(np.float32)/255 + delta[0].permute(1, 2, 0).numpy(), 0, 1)*255).astype(np.uint8)

    @torch.inference_mode()
    def extract(rgb):
        return model.decoder((module.resize_rgb(rgb, 256)*2-1).to(device))[0].cpu().numpy(), 0.

    model.embed = embed
    model.extract = extract
    return model, module.NativeBCH()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', choices=['redmark', 'rosteals', 'sepmark'], required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ['WM_ROOT'] = str(root)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    from watermarking.bench.manifest import payload_for, seed_for
    from watermarking.bench.attack_registry import ATTACKS, plan_attack, apply_attack
    from watermarking.bench.scoring import quality

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    manifest = Path(__file__).resolve().parents[1] / 'image_manifest.json'
    records = [r for r in json.loads(manifest.read_text()) if r['split'] == 'test']
    assert len(records) == 100 and len(ATTACKS) == 28
    if args.limit is not None and not 1 <= args.limit <= 100:
        parser.error('--limit must be between 1 and 100')
    args.out.mkdir(parents=True, exist_ok=False)
    model, ecc = make_model(root, args.model, args.device)
    environment = dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__, device=args.device)
    if args.model == 'redmark':
        import tensorflow as tf
        assert tf.config.list_physical_devices('GPU'), 'TensorFlow GPU unavailable'
        environment['tensorflow'] = tf.__version__
    save(args.out / args.model / 'environment.json', environment)
    records = records[:args.limit] if args.limit else records
    started = time.monotonic()
    for group in ('B', 'A'):
        folder = args.out / f'{args.model}_{group}'
        completed = 0
        for image_index, record in enumerate(records):
            image = np.array(Image.open(root / record['processed']).convert('RGB'))
            assert sha(root / record['processed']) == record['processed_sha256']
            for seed in range(3):
                expected = payload_for(record['image_id'], seed)
                useful = np.unpackbits(np.frombuffer(expected.to_bytes(4, 'big'), dtype=np.uint8))
                capacity = {'redmark': 1024, 'rosteals': 56, 'sepmark': 128}[args.model]
                rng = np.random.default_rng(seed_for(f'maie5421-padding-v1|{record["image_id"]}|{seed}|{args.model}'))
                padding = rng.integers(0, 2, capacity-32, dtype=np.uint8) if group == 'B' else np.zeros(capacity-32, dtype=np.uint8)
                truth = np.r_[useful, padding]
                packet = ecc.encode(truth) if ecc else truth
                stem = f'{record["image_id"]}_s{seed}'
                png = folder / 'embeddings' / (stem+'.png')
                if not png.exists():
                    png.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(model.embed(image, packet)).save(png)
                marked = np.array(Image.open(png).convert('RGB'))
                q = quality(image, marked)
                for attack in ATTACKS:
                    path = folder / 'cases' / f'{stem}_{attack}.json'
                    if path.exists():
                        cached = json.loads(path.read_text())
                        assert cached['image_id'] == record['image_id'] and cached['expected_payload'] == expected
                        completed += 1
                        continue
                    plan = plan_attack(attack, record['image_id'], seed)
                    attacked = apply_attack(marked, plan)
                    scores, threshold = model.extract(attacked)
                    scores = np.asarray(scores)
                    assert np.isfinite(scores).all() and scores.shape == packet.shape
                    bits = scores > threshold
                    decoded, flips = ecc.decode(bits) if ecc else (bits, None)
                    accepted = decoded is not None
                    exact = accepted and np.array_equal(decoded[:32], useful)
                    save(path, dict(variant=f'{args.model}_{group}', image_id=record['image_id'], image_index=image_index,
                         payload_seed=seed, condition=attack, attack_plan=plan, expected_payload=expected,
                         expected_bits=truth.tolist(), packet_bits=packet.tolist(), scores=scores.tolist(), threshold=threshold,
                         decoded_bits=decoded.astype(int).tolist() if accepted else None, corrected_flips=flips,
                         accepted=accepted, exact=bool(exact), wrong=bool(accepted and not exact), erasure=not accepted,
                         raw_payload_ber=float(np.mean(bits[:32] != useful)), raw_payload_exact=bool(np.array_equal(bits[:32], useful)),
                         native_bit_errors=int(np.count_nonzero(bits != packet)), marked_sha256=sha(png),
                         attacked_sha256=hashlib.sha256(attacked.tobytes()).hexdigest(), **q))
                    completed += 1
                save(args.out / f'{args.model}_progress.json', dict(group=group, completed=completed, image=image_index,
                     seed=seed, seconds=time.monotonic()-started))
            # Clean unmarked controls have no payload truth or padding-dependent content.
            negative = folder / 'negatives' / f'{record["image_id"]}.json'
            if not negative.exists():
                scores, threshold = model.extract(image)
                decoded, flips = ecc.decode(scores > threshold) if ecc else (scores > threshold, None)
                save(negative, dict(image_id=record['image_id'], scores=np.asarray(scores).tolist(), threshold=threshold,
                     has_presence_gate=bool(ecc), receiver_returns_payload=decoded is not None, corrected_flips=flips,
                     decoded_bits=decoded.astype(int).tolist() if decoded is not None else None,
                     note='Unconditional payload return cannot be interpreted as a watermark presence detection.'))
            print(args.model, group, image_index+1, '/', len(records), flush=True)
        assert completed == len(records)*3*28
        save(folder / 'status.json', dict(complete=True, positives=completed, negatives=len(records)))
    save(args.out / f'{args.model}_complete.json', dict(complete=True, groups=['A', 'B'], positives=len(records)*168,
         negatives=len(records)*2, seconds=time.monotonic()-started))
    print(args.model, 'COMPLETE', flush=True)


if __name__ == '__main__':
    main()
