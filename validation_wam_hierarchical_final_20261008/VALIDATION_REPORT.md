# Final hierarchical watermark method validation

> Archive navigation note (2026-10-22): this is a historical protocol/report. Script names and machine-local artifacts below describe the original run and may no longer exist in this checkout. See [current commands](../README.md), [migration coverage](../docs/RESEARCH_HISTORY.md), and [中文实验流程](../docs/WORKFLOW_ZH.md). Original measurements and protocol text below are retained.

Date: 2026-10-08. **Decision: MODIFY.**

The tested localized WAM + BCH + CRC-5 channel works well enough to justify a course-project implementation, but the proposed **outer MDS(8,4) layer did not deliver a meaningful exact global-payload recovery advantage over a strong spatial repetition baseline** at equal embedding budget. The final course method should therefore not be presented as MDS-based on this evidence. A simpler WAM + BCH + CRC + spatial repetition route is better supported by this sprint; the course's JPEG/rotation/noise requirements still need their own evaluation.

## Executed evidence

### A. Environment and reproduction

The sprint reused the isolated Python 3.10.14 environment and official [Watermark Anything](https://github.com/facebookresearch/watermark-anything) checkout at commit `2c08af04d037d5667c02f6ddebbda9ff04581c3e`. Its official `wam_mit.pth` file was rehashed as SHA-256 `90ef232384e023bd63245eb0c131abd69d2afc7b8f17a71ccedceb542bf009e2` (377,825,938 bytes). PyTorch 2.5.1, Torchvision 0.20.1, NumPy 1.26.4, Pillow 10.4.0, and scikit-image 0.24.0 ran on macOS arm64 CPU. Upstream git status was clean. All ten natural-image source and processed hashes were rechecked. `results/environment.json` records paths, versions, git status, checkpoint hash, and hashes of earlier case files; exact dependency locks are copied here. No WAM training, fine-tuning, source change, or prior-results edit occurred.

The [protocol](PROTOCOL.md) was frozen before measurement. Decoder inference always received only the marked image or cropped marked pixels. Crop coordinates, original masks, true source symbols, and the unmarked image entered only after decoding for evaluation. All intermediate case results and failures were kept.

### B. CRC-5 definition and packet capacity

The 11 information bits are `physical_packet_id[3] || symbol[8]`, most significant bit first. CRC-5 is the remainder after appending five zero bits and dividing by `g(x)=x^5+x^2+1` (`0x25` including the leading term), with initialization 0, no reflection, and final XOR 0. The transmitted 16-bit pre-BCH packet is `ID[3] || symbol[8] || CRC[5]`. The previously verified primitive BCH(31,16,7) protects that packet; one overall parity bit makes exactly 32 WAM bits. BCH corrects up to three bit errors; the parity bit is counted explicitly within the three-error budget. CRC checks the 16-bit BCH output and rejects a mismatch. There is no confidence threshold.

Each WAM message therefore conveys **one 8-bit coded/data symbol** while spending 3 bits on ID, 5 on integrity, 15 on BCH redundancy, and 1 on overall parity. The local useful symbol rate is **8/32 = 25%**. The final 32-bit global payload occupies eight 32-bit WAM messages, or **32/256 = 12.5%** of transmitted watermark bits. Both outer methods pay exactly the same overhead.

### C. Full packet-codec verification

All **2,048** ID/symbol inputs round-tripped without errors. Because CRC, BCH and parity are linear, exhaustive error-pattern injection into the zero packet characterizes every encoded packet. Every 0–3 bit pattern was exactly repaired: 1 at weight 0, 32 at weight 1, 496 at weight 2, and 4,960 at weight 3. Every **35,960** weight-4 pattern was rejected. Another 7,000 randomized nonzero-input trials confirmed the guarantees.

For errors beyond the guarantee, the complete exhaustive counts were:

| Flipped WAM bits | Patterns | Wrong outputs accepted by BCH alone | Of those, rejected by CRC | Wrong packets still accepted after CRC |
|---|---:|---:|---:|---:|
| 5 | 201,376 | 34,720 | 33,264 | 1,456 |
| 6 | 906,192 | 17,360 | 16,632 | 728 |

CRC rejected **95.8%** of the BCH miscorrections at each weight, but cannot guarantee detection beyond the code radius. Full counts and the specified polynomial are in `results/packet_codec_verification.json`; implementation is in `packet_codec.py`, using the unchanged previous `bch31.py`.

### D. Four-region blind-crop integrity result

This gate reused the previous ten 512×512 natural images, three seeds, four 110×205 corner regions, WAM strength `scaling_w=2.0`, official `multiwm_dbscan` (epsilon 1, min_samples 500, mask threshold 0.5), and the **same six arbitrary-offset crop windows per image/seed**: 90 at ~75% and 90 at ~50% retained area. Each region carried its physical ID 0–3 and a deterministic random 8-bit symbol. Image/seed/crop geometry was paired with the earlier BCH-only sprint, although transmitted bits differ because the payload layout changed.

The 30 clean four-region cases recovered **120/120** packets exactly after BCH/CRC (raw 32-bit exact: 118/120). Mean PSNR was **44.0839 dB**, SSIM **0.988399**; differences from the previous protected four-region means were −0.0086 dB and −0.0000133.

| Blind crop | Regions | Raw WAM exact | Exact after BCH | Exact and CRC-valid | Erasures | Silent wrong accepted |
|---|---:|---:|---:|---:|---:|---:|
| ~75% image retained | 360 | 238 | 306 | 306 | 54 | 0 |
| ~50% image retained | 360 | 132 | 204 | 204 | 156 | 0 |
| Combined | **720** | **370** | **510** | **510** | **210** | **0** |

BCH produced five additional CRC-validity candidates that were actually wrong; CRC rejected **all five**. The 210 remaining failed region cases became explicit erasures. Among physically largely surviving regions (>=75% of the original region retained), **216/225 = 96.0%** returned the correct ID and symbol: 171/179 at ~75% image area and 45/46 at ~50%. The frozen Phase 1 gate passed: 0/720 silent wrong (<=0.5%), >=95% large-region recovery, 100% of failures visible as erasures, and essentially unchanged quality. This is a new payload run; the earlier BCH-only 8/720 silent rate is useful context, not an exact same-bitstream treatment comparison.

### E. Eight-region clean feasibility

Only after Phase 1 passed, the unchanged prior 2×4 grid of eight nonoverlapping 110×205 regions was tested on 10 images × 3 seeds. No outer code was used. Of 30 images, **27 returned all eight exact CRC-valid packets**; the other three returned 4, 5, and 6. Thus **30/30 had at least four valid packets**, with a mean of **7.7**. Across 240 region cases there were **231 exact valid**, **8 erasures**, and **1 silent wrong** (0.42%). Mean detector clusters were 8.03. Mean PSNR was **41.005 dB**, SSIM **0.979835**; mean embedding time was 1.20 s and detection/DBSCAN 2.18 s per image on CPU. All frozen Phase 2 criteria passed.

The three non-perfect eight-region cases were retina seeds 0–2. This image-specific concentration matters: the clean result supports feasibility, but does not prove uniformly reliable eight-region embedding on arbitrary images. A representative clean visualization is `results/eight_clean_visualization.png`.

### F. Exhaustive MDS(8,4) verification

The outer code uses GF(256) with primitive polynomial `0x11d` and the public 8×4 Vandermonde rows at evaluation points 1–8. The four source symbols are the 32-bit global payload. **Every one of the 70 possible four-row subsets was invertible**, and all **70 × 259 = 18,130** tested subset/payload reconstructions were exact. Decoding sorts valid physical packet IDs, solves the first four rows, and checks every additional valid row for consistency. Fewer than four unique accepted IDs, a singular subset, or inconsistency causes failure. No ground-truth symbol is used by the decoder. The matrix and test evidence are in `results/outer_code_verification.json`.

### G. Strong spatial repetition control

The same four source bytes are placed across the eight physical regions as `[A,B,C,D,D,C,B,A]`: duplicate pairs (0,7), (1,6), (2,5), and (3,4) occupy opposite rows and are spatially separated. This uses the same global 32-bit payload, eight region masks, eight 32-bit WAM messages, ID/CRC/BCH packet layout, strength, images, and crop windows as MDS. Repetition accepts a source byte if either copy is CRC-valid; disagreeing valid copies cause global failure. Identical duplicate packet IDs collapse, while conflicting duplicate IDs are erased for both methods. The decoder never consults the true payload.

### H. Final paired MDS versus repetition result

The primary endpoint was exact recovery of **all 32 global bits** after a blind arbitrary-offset crop. Each method ran 30 image/payload cases × six identical crop windows = **180 paired crops**, 90 per retained-area level. The full physical embedding budget was 8 regions × 32 WAM bits for both. The detector saw only the cropped marked image; after WAM localization, BCH, and CRC, the selected outer decoder operated on unique accepted IDs and symbols.

| Image area retained | Spatial repetition | MDS(8,4) | Paired MDS minus repetition |
|---|---:|---:|---:|
| ~75% | **85/90 = 94.44%** | 81/90 = 90.00% | **−4.44 percentage points** |
| ~50% | 76/90 = 84.44% | **81/90 = 90.00%** | **+5.56 percentage points** |
| Combined | 161/180 = 89.44% | 162/180 = 90.00% | **+0.56 percentage points** |

At ~75%, 81 pairs succeeded under both methods, four under repetition only, none under MDS only, and five under neither. At ~50%, 76 pairs succeeded under both, five under MDS only, none under repetition only, and nine under neither. Across the 180 pairs, MDS had only **five wins versus four losses**. All four repetition-only cases occurred on retina, where the MDS-marked versions returned just 1–3 accepted IDs; the five MDS-only cases occurred at ~50% area. This shows the theoretical any-four advantage does not translate into a stable physical-channel gain: embedding different coded bytes changes the WAM error pattern, even at identical region geometry and crop windows.

Clean global recovery was 29/30 for repetition and 30/30 for MDS. Under crop, neither method produced a wrong 32-bit global output; their unsuccessful cases returned an explicit failure. Repetition had one spatially matched CRC-valid wrong local packet among 1,440 crop-region cases (0.069%). MDS had no matched wrong local packet, but one **CRC-accepted unmatched detector cluster**; conservatively treating it as a risky accepted candidate also gives 1/1,440 (0.069%). Both are below the 0.5% gate. No conflicting-ID or repetition-copy disagreement occurred in the final crop set.

The full packet ablation is retained in `results/final_crop_packets.csv` and case-level global results in `results/final_crop_cases.csv`. Among spatially matched packets, raw BER before BCH was 3.17%/4.57% for repetition and 2.70%/3.92% for MDS at ~75%/~50% area respectively; missing clusters are counted separately as erasures, not folded into BER. Exact raw, BCH, and CRC packet counts are in the stage summaries and per-case CSVs. The paired outcome plot is `results/final_comparison.png` (also `.pdf`).

| Method and crop | Mean unique CRC-valid IDs | Raw exact / 720 regions | BCH exact / 720 | CRC-valid exact / 720 | Erased region packets | Matched silent wrong | Raw BER, matched |
|---|---:|---:|---:|---:|---:|---:|---:|
| Repetition, ~75% | 7.00 | 475 | 629 | 629 | 90 | 1 | 3.17% |
| Repetition, ~50% | 5.87 | 370 | 528 | 528 | 192 | 0 | 4.57% |
| MDS, ~75% | 6.89 | 515 | 620 | 620 | 100 | 0 | 2.70% |
| MDS, ~50% | 5.89 | 401 | 530 | 530 | 190 | 0 | 3.92% |

The MDS ~75% set additionally had one CRC-accepted unmatched cluster; it is counted separately from spatially matched region packets and conservatively included in the gate's accepted-risk rate. No method reported a duplicate-ID conflict.

The frozen GO rule required >=10 percentage point combined MDS gain, nonnegative gain at each crop level, comparable quality, and <=0.5% risky accepted packets. **The first two requirements failed.** The observed combined gain was just +0.56 points, and the ~75% condition favored repetition. This is a small screening set of ten images with correlated seeds/crops; the result should not be advertised as a population guarantee. It is sufficient to reject the proposed MDS advantage for method selection under the predeclared gate.

### I. Image quality and capacity

Final MDS images averaged **40.9929 dB PSNR / 0.979896 SSIM**; repetition images averaged **41.0005 dB / 0.979866**. MDS minus repetition was −0.0075 dB and +0.000030 SSIM, far inside the 0.5 dB / 0.005 equivalence screening bounds. Doubling from four to eight marked regions lowered average PSNR from about 44.08 to 41.0 dB and SSIM from about 0.9884 to 0.9799; this is the physical cost of greater coverage, shared by both methods. The net global payload is 32 bits for 256 WAM bits and 68.8% of image pixels inside nominal region masks.

### J. Runtime

On CPU, final embedding averaged **1.245 s/image** for repetition and **1.251 s/image** for MDS. Blind crop WAM detection/DBSCAN averaged **2.59 vs 2.60 s** at ~75% area and **3.55 vs 3.64 s** at ~50%, respectively. Outer decoding averaged about **0.02–0.03 ms** for repetition and **0.28–0.32 ms** for MDS per crop; codec work was negligible relative to WAM. These are observed timings on the current Mac, not hardware-independent throughput promises.

### K. Failure cases and interpretation

- Retina seed 2 in the eight-region clean test yielded five exact packets, three erasures; retina seed 0 had one CRC-valid wrong packet. This indicates the local channel is image dependent even before cropping.
- On four ~75%-area retina crops, repetition recovered the 32-bit global payload while MDS failed with only 1–3 accepted IDs. The paired setup used identical spatial regions and crop windows, so this is a content-dependent embedding/detection difference, not a stronger mathematical property of repetition.
- On five ~50%-area crops, MDS succeeded while repetition failed because at least one of its four source bytes had no valid copy. Those are the cases where MDS's erasure flexibility helped.
- One repetition crop had a partially retained region (70.7%) with 15 raw bit errors; BCH/CRC accepted a wrong packet ID. The global decoder returned failure rather than a wrong payload. One MDS crop had an unmatched CRC-accepted cluster; it did not prevent exact global recovery. Both are retained as local-channel risks.

## Answers and method decision

1. **Does CRC-5 turn severe BCH failures into erasures?** In the exact same new four-region WAM run, five BCH-valid wrong local outputs were rejected by CRC and 0/720 wrong packets remained accepted. Exhaustive 5/6-bit injections showed CRC rejected 95.8% of BCH miscorrections, while allowing some beyond-radius silent errors.
2. **Remaining silent wrong rate?** Four-region crop: 0/720. Eight-region clean: 1/240 (0.42%). Final repetition crop: 1/1,440 spatially matched packet cases (0.069%). Final MDS crop: 0/1,440 matched wrong packets, plus one unmatched accepted cluster (conservative risk 0.069%).
3. **Can eight protected packets coexist?** Yes in this small clean test: 27/30 images returned all eight, 30/30 at least four; mean PSNR 41.01 dB, SSIM 0.9798. Retina remained a clear weak case.
4. **Can any four MDS packets reconstruct the payload?** Yes algebraically: 70/70 subsets invertible, 18,130/18,130 subset/payload tests exact. Physical WAM sometimes supplies fewer than four valid packets.
5. **Does MDS beat strong repetition under equal-budget blind crop?** Only at ~50% area (+5.56 points). It lost at ~75% (−4.44 points); combined gain was +0.56 points, below the frozen +10-point rule.
6. **Exact global-payload improvement?** 162/180 MDS versus 161/180 repetition, a difference of **one paired case out of 180**.
7. **Is quality cost acceptable?** For this screening gate, yes. Both eight-region methods averaged about 41 dB PSNR and 0.98 SSIM, with negligible inter-method difference. This does not establish robustness to other required transformations.
8. **Should the final method use localized WAM + BCH + CRC + outer MDS?** **No.** The measured MDS benefit is too small and inconsistent. Keep localized WAM + BCH + CRC, and use the simpler spatial repetition outer layout as the better-supported course-project candidate. Further robustness evaluation can proceed separately; this sprint did not implement JPEG, rotation, noise, confidence filtering, or training.

## Reproducibility artifacts

`RUNBOOK.md` lists exact stage commands. `results/packet_codec_verification.json` and `results/outer_code_verification.json` contain the independent code checks. `results/phase1_summary.json`, `results/phase2_summary.json`, `results/final_comparison_summary.json`, and `results/final_result.json` contain gate outcomes. The four-region, eight-region, and final paired cases and packets are all in CSV; original stdout/stderr logs and `results/events.jsonl` are retained. The summary figure is available as PNG and PDF.

MODIFY
