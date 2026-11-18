"""Calibration-only resolution diagnostic. All paths supplied by caller."""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from watermarking.bench.adapters.hidden import HiddenAdapter
from watermarking.bench.attack_registry import apply_attack, plan_attack
from watermarking.bench.manifest import load_manifest, payload_for
from watermarking.bench.scoring import quality, score_payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    records = load_manifest(args.root / 'data/images.json', 'calibration')
    assert len(records) == 20
    rows = []
    for protocol in ('native512', 'resize256'):
        adapter = HiddenAdapter(args.root / 'models/hidden', protocol=protocol)
        for strength in (.30, .45):
            adapter.strength = strength
            for rec in records:
                cover = np.array(Image.open(rec['processed']).convert('RGB'))
                payload = payload_for(rec['image_id'], 0)
                marked = adapter.embed(cover, payload).image
                buf = io.BytesIO()
                Image.fromarray(marked).save(buf, format='PNG')
                marked = np.array(Image.open(io.BytesIO(buf.getvalue())))
                q = quality(cover, marked)
                for attack in ('clean', 'jpeg75'):
                    raw = adapter.extract(apply_attack(marked, plan_attack(attack, rec['image_id'], 0)))
                    row = dict(protocol=protocol, strength=strength, image_id=rec['image_id'],
                               attack=attack, **q, **score_payload(raw, payload))
                    rows.append(row)
                print(protocol, strength, rec['image_id'], flush=True)
    with (args.out / 'resolution_smoke.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for protocol in ('native512', 'resize256'):
        for strength in (.30, .45):
            for attack in ('clean', 'jpeg75'):
                subset = [r for r in rows if (r['protocol'], r['strength'], r['attack']) == (protocol, strength, attack)]
                summary.append(dict(protocol=protocol, strength=strength, attack=attack, n=len(subset),
                                    **{k:float(np.mean([r[k] for r in subset])) for k in
                                       ('psnr','ssim','raw_payload_ber','raw_payload_exact','exact','wrong','erasure')}))
    (args.out / 'resolution_smoke_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
