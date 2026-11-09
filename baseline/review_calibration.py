"""Freeze new operating points using calibration data only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from watermarking.bench.adapters.hidden import HiddenAdapter, bits_of
from watermarking.bench.manifest import load_manifest, payload_for
from watermarking.bench.scoring import quality, score_payload
from watermarking.bench.cache import write_json, file_hash
from watermarking.bench.runner import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--config', type=Path, help='Base protocol; default ROOT/configs/protocol.selected.json')
    args = parser.parse_args()
    if (args.out / 'selection_frozen.json').exists():
        raise RuntimeError('Selection already frozen; use the recorded configs')
    torch.set_num_threads(4)
    records = load_manifest(args.root / 'data/images.json', 'calibration')
    summary = json.loads((args.out / 'resolution_smoke_summary.json').read_text())
    def ber(protocol, attack):
        return next(r['raw_payload_ber'] for r in summary if
                    (r['protocol'], r['strength'], r['attack']) == (protocol, .45, attack))
    qualified = ber('resize256', 'clean') < .01 or ber('native512', 'jpeg75') - ber('resize256', 'jpeg75') >= .02
    selected = {'resolution_qualified': qualified, 'formal': []}
    base = json.loads((args.config or args.root / 'configs/protocol.selected.json').read_text())
    base['root'] = str(args.root.resolve())
    base['manifest'] = str((args.root / 'data/images.json').resolve())
    base.pop('calibration_selection', None)
    if qualified:
        model = HiddenAdapter(args.root / 'models/hidden', protocol='resize256')
        residuals = []
        with torch.inference_mode():
            for rec in records:
                cover = np.array(Image.open(rec['processed']).convert('RGB'))
                x = model._tensor(cover)
                low = F.interpolate(x, (256,256), mode='bilinear', align_corners=False, antialias=True)
                z = (low - model.mean) / model.std
                message = torch.tensor([bits_of(0xA5C3,16) + bits_of(payload_for(rec['image_id'],0),32)]).float()*2-1
                delta = (model.encoder(z, message)-z)*model.std
                delta = F.interpolate(delta, x.shape[-2:], mode='bilinear', align_corners=False, antialias=True)
                residuals.append((cover,x,delta))
            def evaluate(strength):
                qualities=[]
                for cover,x,delta in residuals:
                    marked=((x+strength*delta).clamp(0,1)[0].permute(1,2,0)*255).round().byte().numpy()
                    qualities.append(quality(cover, marked))
                return dict(strength=strength, **{k:float(np.mean([q[k] for q in qualities])) for k in ('psnr','ssim')})
            target=39.18387246301725
            lo,hi=.05,1.0
            search=[evaluate(lo),evaluate(hi)]
            if not search[0]['psnr'] > target > search[1]['psnr']:
                write_json(args.out/'calibration_boundary_failure.json',search)
                raise RuntimeError('Prespecified PSNR bracket failed')
            for _ in range(8):
                mid=(lo+hi)/2
                row=evaluate(mid); search.append(row)
                if row['psnr']>target: lo=mid
                else: hi=mid
            best=min(search,key=lambda r:abs(r['psnr']-target))
            selected['hidden_quality_search']=search
            selected['hidden_quality_selected']=best
            for label,strength in [('hidden_resize256_default',.45),('hidden_resize256_psnr',best['strength'])]:
                selected['formal'].append(dict(label=label,method='hidden',config=dict(base,hidden_protocol='resize256',hidden_strength=strength)))
        del model, residuals
    selected['smoke_sha256']=file_hash(args.out/'resolution_smoke.csv')
    for entry in selected['formal']:
        entry['provenance']=provenance(entry['config'],entry['method'])
        write_json(args.out/(entry['label']+'.config.json'),entry['config'])
    write_json(args.out/'selection_frozen.json',selected)
    print('Frozen',[(x['label'],x['method']) for x in selected['formal']],flush=True)


if __name__=='__main__':
    main()
