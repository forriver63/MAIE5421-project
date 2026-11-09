"""Standalone embed/extract interface for the evaluation teammate."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from watermark import Config, embed, extract, to_uint8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    enc = sub.add_parser('embed')
    enc.add_argument('--image', type=Path, required=True)
    enc.add_argument('--message-hex', required=True, help='32 bits = eight hexadecimal characters')
    enc.add_argument('--out', type=Path, required=True, help='New directory')
    enc.add_argument('--alpha', type=float, default=20)
    enc.add_argument('--subband', choices=['HL2', 'HH2'], default='HL2')
    enc.add_argument('--mode', choices=['gray', 'rgb_y'], default='rgb_y')
    dec = sub.add_parser('extract')
    dec.add_argument('--image', type=Path, required=True)
    dec.add_argument('--settings', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'embed':
        raw = bytes.fromhex(args.message_hex)
        if len(raw) != 4:
            parser.error('Use exactly four bytes / eight hex characters')
        with Image.open(args.image) as im:
            image = np.array(im.convert('L' if args.mode == 'gray' else 'RGB'))
        marked, settings, info = embed(image, np.unpackbits(np.frombuffer(raw, dtype=np.uint8)),
                                       Config(alpha=args.alpha, subband=args.subband, color_mode=args.mode))
        args.out.mkdir(parents=True, exist_ok=False)
        Image.fromarray(to_uint8(marked)).save(args.out/'marked.png')
        (args.out/'settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
        print(json.dumps(info))
    else:
        settings = json.loads(args.settings.read_text(encoding='utf-8'))
        mode = 'L' if settings['config']['color_mode'] == 'gray' else 'RGB'
        with Image.open(args.image) as im:
            bits, info = extract(np.array(im.convert(mode)), settings)
        print(json.dumps({'bits': bits.tolist(), 'message_hex': bytes(np.packbits(bits)).hex(), **info}))


if __name__ == '__main__':
    main()
