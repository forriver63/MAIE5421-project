import hashlib
import json
import zipfile
from pathlib import Path
from PIL import Image, ImageOps
from .cache import file_hash, write_json


def seed_for(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big')


def payload_for(image_id, seed):
    return int.from_bytes(hashlib.sha256(f'maie5421-v1|{image_id}|{seed}'.encode()).digest()[:4], 'big')


def prepare(root):
    root = Path(root)
    records = []
    for split, part, ids in [('calibration', 'train', range(1, 21)), ('test', 'valid', range(801, 901))]:
        archive = root / 'downloads' / f'DIV2K_{part}_HR.zip'
        if part == 'train' and (root / 'downloads/DIV2K_train_HR_selected20.zip').exists():
            archive = root / 'downloads/DIV2K_train_HR_selected20.zip'
        with zipfile.ZipFile(archive) as z:
            for i in ids:
                image_id = f'DIV2K_{i:04}'
                member = f'DIV2K_{part}_HR/{i:04}.png'
                source = root / 'data' / 'original' / member
                source.parent.mkdir(parents=True, exist_ok=True)
                if not source.exists():
                    # Only the 20 needed train images are extracted, not all 800.
                    source.write_bytes(z.read(member))
                processed = root / 'data' / 'processed' / f'{image_id}.png'
                processed.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(source) as im:
                    im = ImageOps.fit(ImageOps.exif_transpose(im).convert('RGB'), (512, 512),
                                      method=Image.Resampling.LANCZOS)
                    im.save(processed)
                records.append(dict(image_id=image_id, split=split, source=str(source),
                                    source_sha256=file_hash(source), processed=str(processed),
                                    processed_sha256=file_hash(processed), shape=[512, 512, 3],
                                    provenance='DIV2K official HR; academic research use',
                                    source_url=f'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_{part}_HR.zip'))
    write_json(root / 'data' / 'images.json', records)
    return records


def load_manifest(path, split):
    path = Path(path).resolve()
    records = [r for r in json.loads(path.read_text('utf-8')) if r['split'] == split]
    for record in records:
        for key in ('source', 'processed'):
            if key in record and not Path(record[key]).is_absolute():
                record[key] = str((path.parent / record[key]).resolve())
    return records
