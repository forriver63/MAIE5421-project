"""Compatibility import: authoritative protocol is baselines/shared_protocol.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_protocol import *
