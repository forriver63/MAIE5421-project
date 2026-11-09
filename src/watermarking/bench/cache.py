import gzip
import hashlib
import json
import os
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode('utf-8')
    if path.suffix == '.gz':
        data = gzip.compress(data, mtime=0)
    temporary.write_bytes(data)
    os.replace(temporary, path)


def read_json(path):
    path = Path(path)
    data = path.read_bytes()
    if path.suffix == '.gz':
        data = gzip.decompress(data)
    return json.loads(data)


def load_cache(path, expected_fingerprint):
    if not Path(path).exists():
        return None
    value = read_json(path)  # Truncated/corrupt atomic records are never silently reused.
    if value['fingerprint'] != expected_fingerprint:
        raise ValueError(f'Cache fingerprint mismatch: {path}')
    return value
