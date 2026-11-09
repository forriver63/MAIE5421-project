import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from ..codec.bch import decode_int
from ..codec.packet import packet16


def quality(cover, marked):
    psnr = float(peak_signal_noise_ratio(cover, marked, data_range=255))
    return dict(psnr=psnr if np.isfinite(psnr) else None,
                ssim=float(structural_similarity(cover, marked, data_range=255, channel_axis=2)))


def score_payload(result, expected):
    payload = result.get('payload')
    raw = result.get('raw_payload', payload)
    accepted = result['accepted']
    wrong_bits = None if payload is None else (int(payload) ^ expected).bit_count()
    raw_bits = None if raw is None else (int(raw) ^ expected).bit_count()
    return dict(outcome='exact' if accepted and payload == expected else 'wrong' if accepted else 'erasure',
                exact=int(accepted and payload == expected), wrong=int(accepted and payload != expected),
                erasure=int(not accepted), payload_ber=None if wrong_bits is None else wrong_bits / 32,
                raw_payload_exact=None if raw is None else int(raw == expected),
                raw_payload_ber=None if raw_bits is None else raw_bits / 32,
                nc=None if wrong_bits is None else 1 - wrong_bits / 16)


def score_regions(clusters, labels, expected_words, symbols, oracle_masks, packets, bch_encoded=True):
    """Carrier truth is the actually embedded 32-bit word, including E1 padding."""
    labels = np.asarray(labels)
    overlap = np.zeros((len(clusters), 8), dtype=int)
    for ci, c in enumerate(clusters):
        for ri, mask in enumerate(oracle_masks):
            overlap[ci, ri] = np.count_nonzero((labels == c['label']) & mask)
    assigned = {}
    if clusters:
        rows, cols = linear_sum_assignment(-overlap)
        assigned = {int(r): int(c) for c, r in zip(rows, cols) if overlap[c, r] > 0}
    regions = []
    for ri in range(8):
        row = dict(region=ri, localized=ri in assigned, raw_hamming=None, accepted=False,
                   packet_wrong=None, bch_exact_16=False if bch_encoded else None)
        if ri in assigned:
            ci = assigned[ri]
            c, p = clusters[ci], packets[ci]
            bch = decode_int(c['raw_word'])
            row.update(raw_hamming=(int(c['raw_word']) ^ int(expected_words[ri])).bit_count(),
                       accepted=p.get('selected_for_layout', p['status'] == 'ACCEPTED'),
                       passes_checks=p['status'] == 'ACCEPTED',
                       packet_wrong=(p['packet_id'], p['symbol']) != (ri, symbols[ri]) if p['status'] == 'ACCEPTED' else None,
                       bch_exact_16=(bch['status'] == 'valid' and bch['payload'] == packet16(ri, symbols[ri])) if bch_encoded else None,
                       cluster_label=c['label'], overlap_pixels=int(overlap[ci, ri]))
        regions.append(row)
    return regions, set(assigned.values())
