"""Blind pure decoders. Ground truth and image coordinates are not arguments."""
from itertools import product
from ..codec.bch import decode_int
from ..codec.crc import crc5
from ..codec.outer import collect_accepted_packets, decode_repetition

SINGLE_LAYOUTS = tuple(product((0, 7), (1, 6), (2, 5), (3, 4)))
STRATEGIES = ('A0_full', 'A1_bch_off', 'A2_crc_off') + tuple(f'A3_single_{i:02}' for i in range(16))


def decode_packet_variant(word, *, bch=True, crc=True):
    if not isinstance(word, int) or not 0 <= word < 2**32:
        raise ValueError('Expected unsigned 32-bit carrier')
    correction = decode_int(word) if bch else None
    if bch and correction['status'] != 'valid':
        return dict(status='ERASURE', reason='bch_' + correction['status'], packet_id=None,
                    symbol=None, bch=correction)
    packet = correction['payload'] if bch else word >> 16
    info, check = packet >> 5, packet & 31
    if crc and crc5(info) != check:
        return dict(status='ERASURE', reason='crc_reject', packet_id=None, symbol=None, bch=correction)
    return dict(status='ACCEPTED', reason=None, packet_id=info >> 8, symbol=info & 255, bch=correction)


def decode_words(words, strategy='A0_full'):
    if strategy not in STRATEGIES and strategy != 'E1_no_bch':
        raise ValueError(f'Unknown strategy: {strategy}')
    packets = [decode_packet_variant(int(w), bch=strategy not in ('A1_bch_off', 'E1_no_bch'),
                                     crc=strategy != 'A2_crc_off') for w in words]
    selected = packets
    if strategy.startswith('A3_single_'):
        allowed = SINGLE_LAYOUTS[int(strategy[-2:])]
        selected = [p for p in packets if p['status'] == 'ACCEPTED' and p['packet_id'] in allowed]
    allowed = SINGLE_LAYOUTS[int(strategy[-2:])] if strategy.startswith('A3_single_') else range(8)
    for p in packets:
        p['passes_checks'] = p['status'] == 'ACCEPTED'
        p['selected_for_layout'] = p['passes_checks'] and p['packet_id'] in allowed
    unique, counts = collect_accepted_packets(selected)
    result = decode_repetition(unique)
    source = result['payload']
    payload = None if source is None else int.from_bytes(bytes(source), 'big')
    return dict(strategy=strategy, status=result['status'], accepted=payload is not None, raw_payload=None,
                payload=payload, packets=packets, unique=unique, **counts)
