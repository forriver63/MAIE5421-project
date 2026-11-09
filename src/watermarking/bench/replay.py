"""Re-score saved WAM detections without loading model weights or running inference."""
import argparse
from pathlib import Path
import numpy as np
import torch
from ..masks import make_masks
from .attack_registry import transform_masks
from .cache import file_hash, read_json, write_json
from .decode_variants import STRATEGIES, decode_words
from .runner import code_digest
from .scoring import score_payload, score_regions


def replay(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.drive.lower() != 'd:':
        raise ValueError('Replay requires a distinct D-drive output directory')
    oracle = make_masks(8, torch.device('cpu'))[:, 0].numpy().astype(bool)
    version = code_digest()
    count = 0
    for method in ('full', 'e1'):
        folder = source / method
        if not folder.exists():
            continue
        for manifest in (folder / 'manifest').glob('*.json'):
            write_json(destination / method / 'manifest' / manifest.name, read_json(manifest))
        for path in (folder / 'cases').glob('*.json.gz'):
            case = read_json(path)
            clusters = case['raw']['clusters']
            words = [c['raw_word'] for c in clusters]
            metadata = None
            if not case['negative']:
                metadata = read_json(folder / 'embeddings' / f"{case['image_id']}_s{case['payload_seed']}.json")['metadata']
                labels = np.load(case['raw_masks_path'])['labels']
                masks = transform_masks(oracle, case['attack'], labels.shape)
            results = []
            for strategy in STRATEGIES if method == 'full' else ('E1_no_bch',):
                result = decode_words(words, strategy)
                if metadata is not None:
                    result.update(score_payload(result, case['expected_payload']))
                    regions, assigned = score_regions(clusters, labels, metadata['words'], metadata['symbols'],
                                                       masks, result['packets'], bch_encoded=method == 'full')
                    result['regions'] = regions
                    result['unmatched_accepted_clusters'] = sum(i not in assigned and p.get('selected_for_layout', p['status'] == 'ACCEPTED')
                                                               for i, p in enumerate(result['packets']))
                    result['wrong_accepted_clusters'] = sum(p.get('selected_for_layout', p['status'] == 'ACCEPTED') and
                        p['symbol'] != metadata['symbols'][p['packet_id']] for p in result['packets'])
                results.append(result)
            case.update(results=results, replay_source=str(path), replay_source_sha256=file_hash(path),
                        analysis_version=version)
            write_json(destination / method / 'cases' / path.name, case)
            count += 1
    write_json(destination / 'replay_provenance.json', dict(source=str(source), analysis_version=version,
                                                            cases=count, model_inference_calls=0))
    print('Replayed', count, 'cases; model inference calls: 0')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    replay(args.run_dir, args.out)
