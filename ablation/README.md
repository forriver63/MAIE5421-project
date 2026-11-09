# Component ablation

This experiment evaluates the final eight-region WAM system with 32 useful bits. It is separate from the earlier outer-code experiment in `experiments/ablation/`, which remains unchanged.

| Variant | Change | Core exact recovery |
|---|---|---:|
| A0 Full | BCH + parity, CRC-5 and two spatial copies | 78.34% |
| A1 | Bypass BCH correction and validity rejection on the same codewords | 56.50% |
| A2 | Disable CRC filtering; keep the BCH validity gate | 76.50% |
| A3 | One copy per byte; average all 16 fixed layouts | 58.05% |
| E1 | Remove BCH at embedding; retain packet layout, CRC and repetition | 57.20% |

Full-minus-variant differences are 21.85 pp for A1 (95% paired CI 19.98–23.77), 1.85 pp for A2 (1.34–2.44), 20.30 pp for A3 (18.58–22.18), and 21.14 pp for E1 (19.22–23.06). A1 and E1 provide complementary evidence for BCH. A1 does not undo the redundancy embedded in the image; E1 changes the embedded carrier, so their interpretations differ.

CRC chiefly improves reliability: core wrong-message output falls from 0.6574% without CRC to 0.02546% with CRC. Rejection counts as failure. A3 selects packets by predicted identifiers, never ground-truth region identities. Ground-truth masks are used only for diagnostic region scoring.

## Reproduce

Follow [the shared setup](../baseline/README.md#setup-and-inputs), then run from the repository root:

```sh
python ablation/run.py --config configs/protocol.local.json --run-dir output/components --stage decode
python ablation/run.py --config configs/protocol.local.json --run-dir output/components --stage decode --negative
python ablation/run.py --config configs/protocol.local.json --run-dir output/components --stage encode
python -m watermarking.bench.finalize --run-dir output/components --out output/components-summary
```

Full uses 8,400 positive cases and 100 clean negative images. All 19 decoding strategies reuse each Full detection: A0, A1, A2 and 16 A3 layouts. E1 uses the 6,000 core positive cases; its unmarked receiver controls reuse the Full/BCH-off detections. Do not describe these as independent E1 negative images.

The core score gives equal weight to six attack families, averaging severities and seeds within image first. It is not the flat average over all 28 attacks. The eight extended attacks are reported separately. Bootstrap resamples 100 images, not individual messages or layouts, 2,000 times with seed 20261106.

Small summary tables are in `results/`; complete large caches remain outside Git. [The combined report](../baseline/REPORT_EN.html) includes bit-level coverage, failure analysis. New results never overwrite the historical validation directories.
