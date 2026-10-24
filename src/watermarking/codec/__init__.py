"""Error-control codecs (pure Python, no torch or image dependencies).

These are migrated verbatim from the historical sprints so that the bit-level
behavior is identical:

- ``bch``     : extended BCH(31, 16) with one overall parity bit (from bch31.py)
- ``crc``     : CRC-5 polynomial 0x25 (from packet_codec.py)
- ``packet``  : 16-bit packet = 3-bit ID + 8-bit symbol + 5-bit CRC
- ``outer``   : spatial repetition and GF(256) Vandermonde MDS outer codes
"""

from . import bch, crc, outer, packet

__all__ = ["bch", "crc", "packet", "outer"]
