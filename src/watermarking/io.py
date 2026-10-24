"""Shared file / hash / CSV / JSON / event-log helpers.

Deduplicated from the historical sprints (run_validation.py, run_sprint.py,
run_phase1.py, run_robustness.py all carried near-identical copies).
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def log_event(path, kind: str, **fields) -> dict:
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": kind, **fields}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, default=str) + "\n")
    print(json.dumps(record, default=str), flush=True)
    return record
