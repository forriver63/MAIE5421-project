# Official WAM inner-ECC validation sprint

> Archive navigation note (2026-10-22): this is a historical protocol/report. Script names and machine-local artifacts below describe the original run and may no longer exist in this checkout. See [current commands](../README.md), [migration coverage](../docs/RESEARCH_HISTORY.md), and [中文实验流程](../docs/WORKFLOW_ZH.md). Original measurements and protocol text below are retained.

Date: 2026-10-08. Decision: **MODIFY**.

## Executed evidence

### A. Environment and reproducibility

The experiment reused the isolated Python 3.10.14 Conda environment, official [Watermark Anything](https://github.com/facebookresearch/watermark-anything) commit `2c08af04d037d5667c02f6ddebbda9ff04581c3e`, and official `wam_mit.pth` checkpoint from the prior validation. The checkpoint SHA-256 was rechecked as `90ef232384e023bd63245eb0c131abd69d2afc7b8f17a71ccedceb542bf009e2` (377,825,938 bytes). PyTorch 2.5.1, Torchvision 0.20.1, NumPy 1.26.4, Pillow 10.4.0, and scikit-image 0.24.0 ran on macOS arm64 CPU. The upstream git checkout was clean. The exact Python path, git status, package versions, checkpoint path/hash, and previous-result hash are in `results/environment.json`; dependency locks are copied into this directory. All ten processed image hashes were verified before use. No WAM source, checkpoint, or prior result file was modified. No training was performed.

The [frozen protocol](PROTOCOL.md) was written before WAM measurements. Decoder inputs were marked images or cropped marked images alone. Original masks, crop coordinates, and true messages were used only after inference for scoring.

### B. BCH codec verification

The manually implemented codec is a genuine primitive narrow-sense BCH(31,16) with designed distance 7 and correction radius 3. Over GF(32), using primitive field polynomial `x^5+x^2+1`, the generator was computed from the distinct cyclotomic cosets of roots alpha^1 through alpha^6: `{1,2,4,8,16}`, `{3,6,12,24,17}`, `{5,10,20,9,18}`. Its degree is 15 and binary representation is `0x8faf`. This construction follows the standard narrow-sense BCH generator rule described in [MathWorks' BCH documentation](https://www.mathworks.com/help/comm/ref/bchgenpoly.html). The implementation independently verified that the generator divides `x^31+1`.

One even-parity bit extends the code to 32 transmitted bits. The 31-bit syndrome decoder corrects at most three BCH-bit errors; parity is separately repaired only if the **combined** correction count stays at most three. A four-bit pattern is rejected, not corrected. Parity adds detection distance; it does not raise the guaranteed correction radius. Exhaustive enumeration of all 65,535 nonzero codewords found minimum distance 7 before parity and 8 after parity. All 65,536 clean payloads round-tripped. Every 1-bit (32), 2-bit (496), 3-bit (4,960), and 4-bit (35,960) error pattern on the zero codeword was checked: 1–3 corrected, 4 rejected. Linearity extends that check to every codeword. Another 7,000 random 0–6-bit trials passed the corresponding guarantees. For 10,000 random 5-bit and 10,000 random 6-bit errors, the decoder had 1,767 and 212 silent miscorrections respectively; five or more errors are outside the code's guarantee. Full counts: `results/codec_verification.json`.

**Capacity cost:** one region carries 16 useful bits in 32 WAM bits: 50% useful rate. Relative to a raw 32-bit message, inner coding spends 16 bits on 15 BCH checks and one overall parity bit. This is an intentional robustness/capacity tradeoff; raw 32-bit capacity and protected 16-bit capacity are not equivalent.

### C. Four-region clean paired results

The clean test reused exactly ten natural sample images, three seeds, the same four 110×205 pixel corner regions, 512×512 images, WAM strength `scaling_w=2.0`, and official `multiwm_dbscan` (`epsilon=1`, `min_samples=500`, mask threshold 0.5). Pairing is by image and seed; message bits differ because this run carries protected 16-bit payloads.

| Metric | Prior uncoded run | This protected run |
|---|---:|---:|
| Cases / local packets | 30 / 120 | 30 / 120 |
| Raw 32-bit packets exact | 112/120 (93.33%) | 115/120 (95.83%) |
| Exact useful local payloads after BCH | Not applicable | **120/120 (100%)** |
| Images with all four useful payloads exact | 22/30 raw messages (73.33%) | **30/30 (100%)** |
| Raw BER among matched packets | 0.4167% | 0.1823% |
| Detected clusters, mean | 4.00 | 4.00 |
| Localization union IoU, mean | 0.9443 | 0.9449 |
| PSNR, mean | 44.0838 dB | 44.0925 dB |
| SSIM, mean | 0.988414 | 0.988412 |

On the **same new marked images**, raw exact recovery rose from 115/120 to 120/120 after BCH decoding. Four packets had one wrong raw bit and one had three wrong raw bits; all five were corrected. There were no clean decoder failures or silent wrong useful messages. The predeclared Phase 2 gate passed all four criteria: >=98% useful packet recovery, >=90% all-four-exact images, <=1% silent wrong decodes, and no meaningful PSNR/SSIM loss. The old versus new raw rates are contextual because the embedded messages differ; the new-run raw versus new-run protected comparison isolates the ECC effect.

### D. Blind arbitrary-offset crop localization

Phase 3 ran because Phase 2 passed. It tested the same 30 four-message marked image/seed cases with three fixed random offsets per retained-area level, including square, 4:3, and 3:4 crops: 90 crops at about 75% area and 90 at about 50%, 720 original region cases total. The detector saw only each cropped marked image. Region geometry and crop offsets were introduced **after** detection and BCH decoding to assess physical survival and spatial assignment.

| Metric | ~75% image retained | ~50% image retained |
|---|---:|---:|
| Crops | 90 | 90 |
| Mean localized clusters | 3.81 | 3.00 |
| Raw exact 32-bit messages / 360 regions | 241 (66.94%) | 134 (37.22%) |
| Exact useful messages after BCH / 360 regions | 309 (85.83%) | 215 (59.72%) |
| Largely surviving regions (>=75% region retained) | 179 | 46 |
| Exact among largely surviving | 173/179 (96.65%) | 46/46 (100%) |
| Silent wrong useful decodes | 3 | 5 |
| Decoder-visible missing clusters | 18 | 94 |
| Localized but BCH-rejected messages | 30 | 46 |

Across both levels, 608/720 region identities were spatially localized, 375 raw messages were exact, and BCH raised exact useful recovery to 524. The matched raw bit error rate was 4.54% under crop. Among 196 failed useful-payload cases, 112 had no localized cluster, 76 had a localized cluster rejected by BCH/parity, and **8 were accepted but wrong**. Those eight silent errors are 1.11% of all 720 region cases and 1.50% of 532 decoder-accepted packets. Every silent case had at least five wrong raw bits and occurred in a partially retained region (retention 2.7%–57.3%); none occurred among the 225 largely surviving regions. Still, three of the eight occurred with ~75% total image area retained.

Of the 196 failed region cases, 74 were fully absent and 116 more retained less than 75% of their original area. Six failures occurred despite >=75% region retention, all at ~75% image area; two of those regions were fully retained yet had 4 or 6 raw bit errors. Thus physical truncation explains most failures, but not all of the channel behavior. The decoder can convert many corrupted packets into erasures, yet it occasionally emits a plausible wrong packet.

The predeclared Phase 3 gate required >=75% exact recovery for largely surviving regions at each level, >=80% of crops with at least one cluster, <=1% silent wrong local decodes, and a majority of failed packets with <75% region survival. The first, second, and fourth criteria passed. The silent wrong rate was **8/720 = 1.11%**, so the gate failed and the sprint stopped here. `results/crop_cases.csv` and `results/crop_packets.csv` retain all cases, including failures.

### E. Eight-region clean feasibility

**NOT RUN.** Phase 3 did not pass. There is no executed evidence from this sprint that eight BCH-protected WAM messages can coexist at acceptable quality or yield at least four valid packets.

### F. Outer MDS versus repetition

**NOT RUN.** The eight-region prerequisite was not reached. There is no end-to-end, fair 32-bit global-payload comparison for MDS and optimized repetition in this sprint. The prior coding-only pilot remains separate evidence about algebra, not WAM cropping performance.

### G. Failure examples

- A 50% crop of astronaut, seed 0, left-top region retained 21.8% of its area. WAM localized a cluster with six wrong raw bits; BCH accepted a wrong 16-bit value. This is a silent miscorrection beyond its three-bit guarantee. See `results/blind_crop_50pct.png`.
- A 75% crop of Hubble, seed 2, one region retained 10.0%; WAM localized a low-confidence cluster with 15 wrong bits, and BCH accepted a wrong value.
- A 75% crop of retina, seed 2, included two fully retained regions with 4 and 6 wrong raw bits, both rejected by BCH. Crop processing can therefore damage decoding even when the whole region remains visible.

### H. Image quality and visuals

The protected clean images averaged **44.0925 dB PSNR** and **0.988412 SSIM**, changes of +0.0087 dB and −0.0000013 relative to the prior run. These tiny observed differences are within the frozen gate; they do not prove formal image-quality equivalence across all content. `results/clean_four_regions.png` shows a representative clean embed/detection. `results/blind_crop_50pct.png` shows a reconstructed crop example with the decoder input and scoring-only masks clearly separated. `results/summary_figure.png` and `.pdf` summarize packet recovery by channel condition and physical region retention.

### I. Runtime

On CPU, the clean four-region run averaged **0.615 s embedding** and **0.955 s WAM detection/DBSCAN** per image. ECC encoding averaged **0.035 ms** for four packets and decoding **0.018 ms** per image. Crop WAM detection averaged **0.994 s** at 75% area and **0.888 s** at 50% area; ECC decode averaged about **0.02 ms** per crop. The codec cost is negligible relative to WAM inference in this environment.

### J. Final method decision and answers

1. **Does BCH repair the observed clean 1–3 bit errors?** Yes. This run had five bad raw packets (four with one error, one with three), and all five were recovered exactly. The mathematical and injection tests verify a three-bit correction guarantee.
2. **What is the before/after useful recovery?** On the same clean protected bitstreams, raw 32-bit exact was 115/120 (95.83%); post-BCH 16-bit useful exact was 120/120 (100%). Under blind crop, the analogous counts were 375/720 raw exact and 524/720 useful exact. The exactness criteria cover 32 versus 16 bits, with the capacity cost stated above.
3. **What capacity is spent?** Each 32-bit WAM message carries 16 useful bits, a 50% useful rate; 15 bits are BCH redundancy and one bit is parity.
4. **Does crop become mostly missing packet rather than corrupted packet?** Mostly decoder-visible erasure: 188/196 failed cases were missing or rejected. But eight wrong packets were accepted as valid (1.11% of all region cases), so this property is not reliable enough under the frozen gate. Only 74/196 failed regions were physically removed completely.
5. **Can eight protected packets coexist at acceptable quality?** Unknown; Phase 4 was not reached.
6. **Does MDS beat optimized repetition at equal physical budget?** Unknown; Phase 5 was not reached.
7. **What should the final course-project method be based on executed evidence?** Keep WAM plus a small BCH-style inner code as a promising *local clean-image component*. Do not commit to the full localized-WAM + outer-MDS method. The next bounded change, if this candidate is revisited, is an explicit packet-integrity mechanism within the 16-bit useful field (with its payload cost counted) and a new blind-crop gate; the present evaluation must not be reclassified as a pass.

## Interpretation and unexecuted future ideas

The clean bottleneck from the previous sprint was substantially repaired at half the local useful rate. Under cropping, WAM often decodes even partly retained regions, and nearly all largely surviving regions were recovered. The residual risk is false acceptance after five or more raw bit errors. The unused Phase 5 packet proposal reserved five of 16 pre-BCH bits for integrity; such a check might reject many false accepts, but it was **not implemented or measured here**. Eight-region feasibility and end-to-end MDS benefit remain open. Optional confidence margins correlated with raw error count (Spearman rho −0.78 over matched crop packets), but no threshold was tuned or used, and this correlation is exploratory.

## Reproducibility artifacts

`RUNBOOK.md` lists executed commands. `results/environment.json`, `results/codec_verification.json`, `results/clean_cases.csv`, `results/clean_packets.csv`, `results/paired_comparison.csv`, `results/crop_cases.csv`, `results/crop_packets.csv`, `results/clean_summary.json`, `results/crop_summary.json`, `results/final_result.json`, `results/events.jsonl`, phase logs, and the figures contain the full executed record. The decoder and evaluation implementation are in `bch31.py`, `run_sprint.py`, and `run_crop.py`.

MODIFY
