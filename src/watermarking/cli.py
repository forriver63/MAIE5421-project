"""Command-line interface for embedding and detecting the 32-bit watermark."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import sys

from PIL import Image

from . import __version__
from .pipeline import detect_image, embed_image


def _parse_payload(text: str) -> int:
    value = int(text, 0)
    if not 0 <= value < (1 << 32):
        raise SystemExit("payload must be a 32-bit unsigned integer")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="wm", description="Localized image watermarking")
    parser.add_argument("--version", action="version", version=f"wm {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    embed = sub.add_parser("embed", help="embed a 32-bit payload into an image")
    embed.add_argument("input", help="input image path")
    embed.add_argument("output", help="watermarked output image path")
    embed.add_argument("--payload", required=True, help="32-bit payload (decimal or 0x...)")

    detect = sub.add_parser("detect", help="recover the payload from an image")
    detect.add_argument("input", help="image path")
    detect.add_argument("--json", action="store_true", help="print machine-readable JSON")

    args = parser.parse_args(argv)
    if args.command == "embed":
        image = Image.open(args.input)
        result = embed_image(image, _parse_payload(args.payload))
        result.save(args.output)
        print(f"embedded payload into {args.output}")
    elif args.command == "detect":
        # The upstream model loader prints diagnostics; reserve stdout for the
        # CLI response so --json remains parseable on a cold model load.
        with redirect_stdout(sys.stderr):
            result = detect_image(Image.open(args.input))
        if args.json:
            import json
            print(json.dumps(result))
        else:
            print(f"status={result['status']} payload={result['payload']} "
                  f"valid_ids={result['valid_ids']} clusters={result['detected_clusters']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
