# Final WAM robustness validation

> Archive navigation note (2026-10-22): this is a historical protocol/report. Script names and machine-local artifacts below describe the original run and may no longer exist in this checkout. See [current commands](../README.md), [migration coverage](../docs/RESEARCH_HISTORY.md), and [中文实验流程](../docs/WORKFLOW_ZH.md). Original measurements and protocol text below are retained.

Executed on 2026-10-08. The frozen candidate is localized official WAM with 8 regions, BCH(31,16) t=3 plus one overall parity bit, CRC-5, and spatial repetition `[A,B,C,D,D,C,B,A]`. Four source bytes form the 32-bit useful global payload. Each region embeds 3 ID bits + 8 data bits + 5 CRC bits through a 32-bit WAM word. No model training, source modification, new ECC, or MDS was used in this sprint.

## Environment and reproducibility

The exact official source commit, checkpoint SHA-256, package versions, Python path, and hashes of all ten public natural images are in [environment.json](results/environment.json). The executable protocol was fixed in [PROTOCOL.md](PROTOCOL.md) before measurement. Executed commands are collected in [REPRODUCE.sh](REPRODUCE.sh). Raw logs, case CSV, packet CSV, crop geometry, machine-readable summary, and artifact hashes are in `logs/` and `results/`. The official checkpoint hash is `90ef232384e023bd63245eb0c131abd69d2afc7b8f17a71ccedceb542bf009e2`; source commit `2c08af04d037d5667c02f6ddebbda9ff04581c3e`.

## Robustness results

Each attack used the same 10 images × 3 deterministic global payload seeds, with 8 packets per marked image. The decoder received only attacked pixels. The original payload and transformed masks were used after inference for scoring. JPEG used Pillow RGB JPEG, 4:2:0 subsampling; noise σ is on the 0–255 scale; rotations used Pillow bicubic, fixed 512×512 canvas, black fill.

| Attack | Exact global | Clusters | Valid IDs | Erasures | Raw BER¹ | BCH repaired packets | Silent wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 29/30 (96.7%) | 8.07 | 7.73 | 3.3% | 0.97% | 19 | 0 |
| jpeg75 | 24/30 (80.0%) | 7.93 | 6.67 | 16.7% | 4.34% | 79 | 0 |
| jpeg50 | 12/30 (40.0%) | 8.27 | 3.80 | 52.5% | 13.12% | 83 | 0 |
| noise2 | 30/30 (100.0%) | 8.20 | 7.70 | 3.8% | 1.02% | 22 | 0 |
| noise5 | 28/30 (93.3%) | 8.07 | 7.63 | 4.6% | 1.23% | 29 | 0 |
| rotate-1 | 28/30 (93.3%) | 8.30 | 7.43 | 7.1% | 2.73% | 65 | 0 |
| rotate+1 | 28/30 (93.3%) | 7.93 | 7.30 | 8.8% | 2.49% | 43 | 0 |
| rotate-2 | 27/30 (90.0%) | 8.07 | 7.37 | 8.3% | 2.80% | 59 | 0 |
| rotate+2 | 27/30 (90.0%) | 8.13 | 7.17 | 10.4% | 2.91% | 43 | 0 |
| rotate-5 | 27/30 (90.0%) | 7.90 | 7.07 | 11.7% | 3.21% | 53 | 0 |
| rotate+5 | 26/30 (86.7%) | 8.10 | 7.10 | 11.7% | 3.44% | 61 | 0 |
| rotate-10 | 6/30 (20.0%) | 7.73 | 2.90 | 63.7% | 14.60% | 76 | 0 |
| rotate+10 | 3/30 (10.0%) | 7.33 | 2.33 | 70.8% | 16.05% | 61 | 0 |
| crop75_jpeg75 | 12/30 (40.0%) | 7.67 | 4.10 | 48.8% | 11.82% | 86 | 1 |

¹ Raw BER is computed only among spatially matched detected packets; lost packets are counted as erasures. BCH repaired means a packet had at least one raw bit error and its exact intended packet survived BCH+CRC. Case-level and packet-level evidence: [cases.csv](results/cases.csv), [packets.csv](results/packets.csv). [Recovery and rotation figure](figures/global_recovery_and_rotation.png); [representative images](figures/representative_attacks.png).

## Clean reference and image quality

The exact 32-bit payload was recovered in 29/30 clean images. Mean CRC-valid IDs: 7.73/8; exact valid local packets: 232/240; silent wrong packets: 0. Mean marked-image PSNR 41.00 dB and SSIM 0.97987. These are quality measurements before attacks.

## JPEG and Gaussian noise

JPEG Q75 recovered 24/30 globally; Q50 12/30. Noise σ2 recovered 30/30; σ5 28/30. The table separates raw matched-bit errors from BCH-corrected packets, CRC-valid IDs, and erasures. The exact rates define the observed robustness; this small sample does not establish a population guarantee.

JPEG Q50 still yielded 8.27 localized clusters on average, but only 3.80 unique CRC-valid IDs; this is predominantly packet corruption/rejection rather than complete failure to find regions. BCH repaired 83 exact packets, yet the remaining loss often removed both copies of a source byte. The combined crop+JPEG condition likewise yielded 7.67 clusters but only 4.10 valid IDs, with one spatially matched silent wrong packet. The case-level decoder produced no wrong complete 32-bit outputs in this sample.

## Rotation and conditional search

Without correction, the pooled ±1° and ±2° recovery rate was 110/120 (91.7%). The predeclared search trigger was **not met**. Not executed: the predeclared small-angle failure trigger did not fire.

## Combined crop + JPEG

The first fixed arbitrary-offset ~75%-area crop per case followed by JPEG Q75 recovered 12/30 (40.0%). Crop coordinates were evaluation-only. This is a small composition check, distinct from the prior 180-crop-only paired study (repetition 85/90 at 75% and 76/90 at 50%).

## Negative control

On 10 unmarked public natural images, the decoder produced 0 clusters in total, 0 CRC-valid clusters, and 0 complete false 32-bit outputs. This is a sanity check, not a calibrated false-positive probability.

## Runtime

Mean WAM embedding 1.23 s per 8-region image; mean clean detection 2.16 s; mean clean packet decoding 0.031 ms; mean clean encoder 0.061 ms. Attack-specific detection times are in the robustness table's machine-readable summary.

## Evidence chain and method selection

- Earlier raw four-region WAM: 112/120 exact 32-bit packets, with all 8 errors limited to 1–3 bits and all failed packets localized.
- Earlier BCH clean paired test: 115/120 raw exact versus 120/120 protected 16-bit useful payloads exact, at half the raw WAM message capacity.
- Earlier BCH+CRC four-region blind crop: 510/720 exact accepted packets, 0 silent wrong; 216/225 substantially surviving regions recovered.
- Earlier eight-region feasibility: 30/30 had at least four valid packets, mean 7.7/8, PSNR 41.01 dB, SSIM 0.97984.
- Earlier paired outer-code study: repetition 161/180 and MDS 162/180 exact cropped global payloads. The one-case overall difference did not justify MDS; repetition remains frozen.

The physical embedding budget is eight 32-bit WAM messages (256 embedded bits) for one 32-bit global useful payload: an 8:1 bit ratio. Each local 32-bit word carries an 8-bit data symbol alongside 3 ID, 5 CRC, 15 BCH redundancy, and 1 parity bit; each of the four data symbols is spatially repeated. Mean clean PSNR/SSIM quantify the visual cost. This budget favors recovery over capacity.

The final decision is **LIMITED** because JPEG Q50 and crop+JPEG Q75 each recovered only 12/30 global payloads, while ±10° rotation recovered only 6/30 and 3/30. Mild Gaussian noise and rotations through ±5° remain useful, but the tested compression and combined attack do not support a general robust-watermark claim. No further module was added.

## Failure cases and limitations

- `06_2` under `clean`: missing_source_symbol; 6 valid IDs, 2 region erasures, 0 silent wrong packets.
- `00_1` under `jpeg75`: missing_source_symbol; 6 valid IDs, 2 region erasures, 0 silent wrong packets.
- `00_2` under `jpeg75`: missing_source_symbol; 6 valid IDs, 2 region erasures, 0 silent wrong packets.
- `00_1` under `jpeg50`: missing_source_symbol; 3 valid IDs, 5 region erasures, 0 silent wrong packets.
- `00_2` under `jpeg50`: missing_source_symbol; 5 valid IDs, 3 region erasures, 0 silent wrong packets.
- `06_0` under `noise5`: missing_source_symbol; 5 valid IDs, 3 region erasures, 0 silent wrong packets.
- `06_2` under `noise5`: missing_source_symbol; 5 valid IDs, 3 region erasures, 0 silent wrong packets.
- `06_0` under `rotate+2`: missing_source_symbol; 2 valid IDs, 6 region erasures, 0 silent wrong packets.
- `06_1` under `rotate+2`: missing_source_symbol; 2 valid IDs, 6 region erasures, 0 silent wrong packets.
- `00_0` under `rotate+10`: missing_source_symbol; 0 valid IDs, 8 region erasures, 0 silent wrong packets.
- `00_1` under `rotate+10`: missing_source_symbol; 1 valid IDs, 7 region erasures, 0 silent wrong packets.
- `00_0` under `crop75_jpeg75`: missing_source_symbol; 3 valid IDs, 5 region erasures, 0 silent wrong packets.
- `00_1` under `crop75_jpeg75`: missing_source_symbol; 3 valid IDs, 5 region erasures, 0 silent wrong packets.
- `09_1` under `crop75_jpeg75`: repetition_conflict; 5 valid IDs, 3 region erasures, 1 silent wrong packets.

Rotation uses a fixed-size black-filled canvas, so boundary clipping and interpolation are part of the tested attack. Search, if executed, is limited to its nine fixed candidate angles and was assessed only on ±1°/±2°; no general geometric invariance is claimed. The 10-image collection and 3 seeds are a course-project screening set. BER conditioned on detected packets omits full packet loss by design; the erasure column complements it. CRC-5 and a small negative control cannot prove a low false-positive rate against arbitrary imagery or adversarial attacks. No claims are made for attacks not executed here.

## Final questions

1. **JPEG:** Q75 24/30, Q50 12/30; limited at the tested quality levels.
2. **Noise:** σ2 30/30, σ5 28/30; sufficient within these tested settings.
3. **Rotation degradation:** ±1°/±2° pooled 110/120; exact per-angle values are in the table and rotation curve.
4. **Search necessary?** No, the predeclared failure gate did not fire.
5. **Blind search recovery:** Not tested because it was not indicated by the gate.
6. **Crop + JPEG:** 12/30 exact under the bounded ~75% + Q75 check.
7. **Unmarked false global outputs:** 0/10.
8. **Trade-off:** 32 useful global bits per 256 embedded WAM bits, with clean PSNR 41.00 dB and SSIM 0.97987.
9. **Final method:** Official localized WAM, fixed BCH(31,16)+parity, CRC-5, and spatial repetition; no additional module.
10. **Known limitations:** see failure cases above; the measured attack conditions and finite sample bound all conclusions.

LIMITED
