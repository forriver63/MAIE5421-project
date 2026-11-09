"""Copy only the ten committed team input PNGs, without regenerating them."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
from protocol import IMAGE_NAMES, TEAM_COMMIT, TEAM_URL, crop_box, source_bits, source_bytes

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(reference=None):
    reference = Path(reference) if reference else ROOT.parents[1]
    # Branch commits may advance: pin data content, not the developer's HEAD.
    commit = TEAM_COMMIT
    subprocess.check_call(['git', '-C', str(reference), 'cat-file', '-e', commit+'^{commit}'])
    src = reference/'validation_wam_20261007/official_results/inputs'
    destination = ROOT/'data/team10'
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for index, name in enumerate(IMAGE_NAMES):
        source, target = src/f'{name}.png', destination/f'{name}.png'
        with Image.open(source) as im:
            if im.size != (512, 512) or im.mode != 'RGB':
                raise ValueError(f'Unexpected input: {name} {im.size} {im.mode}')
        digest = sha256(source)
        pinned = subprocess.check_output(['git', '-C', str(reference), 'show',
                                          commit+':'+str(source.relative_to(reference))])
        if hashlib.sha256(pinned).hexdigest() != digest:
            raise RuntimeError(f'Source differs from pinned dataset: {source}')
        if target.exists():
            if sha256(target) != digest:
                raise RuntimeError(f'Refusing to overwrite modified input {target}')
        else:
            shutil.copyfile(source, target)
        records.append({'image_index': index, 'name': name, 'file': f'team10/{name}.png',
                        'sha256': digest, 'shape': [512,512,3],
                        'source_path': str(source.relative_to(reference))})
    manifest = {'repo': TEAM_URL, 'commit': commit, 'images': records,
                'preprocessing': 'byte-identical copies of committed RGB 512x512 PNGs; no new resize/crop'}
    (ROOT/'data/manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    payloads = []
    for i, name in enumerate(IMAGE_NAMES):
        for seed in range(3):
            source = source_bytes(i, seed)
            payloads.append({'case_id': f'{i:02d}_{seed}', 'image': name, 'image_index': i,
                             'payload_seed': seed, 'source_bytes': source.tolist(),
                             'payload_hex': bytes(source).hex(), 'bits_msb_first': source_bits(i, seed).tolist(),
                             'crop75_xywh': list(crop_box(i, seed))})
    (ROOT/'data/payloads.json').write_text(json.dumps(payloads, indent=2), encoding='utf-8')
    print(f'Prepared and hash-verified {len(records)} RGB images and {len(payloads)} 32-bit cases.')


if __name__ == '__main__':
    prepare()
