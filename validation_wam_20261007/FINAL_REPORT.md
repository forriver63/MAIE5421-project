# Official WAM minimal validation report

> Archive navigation note (2026-10-22): this is a historical protocol/report. Script names and machine-local artifacts below describe the original run and may no longer exist in this checkout. See [current commands](../README.md), [migration coverage](../docs/RESEARCH_HISTORY.md), and [中文实验流程](../docs/WORKFLOW_ZH.md). Original measurements and protocol text below are retained.

Date: 2026-10-07

# NO-GO

**Localized decoding is not sufficiently reliable in the clean four-region test, so the proposed localized 4-of-8 erasure-coded watermark method should not be adopted on the evidence from this sprint.**

This is a decision about the proposed configuration and validation gate, not a claim that WAM is unusable. WAM localized all tested clean regions well, but exact 32-bit message decoding failed often enough that the predeclared clean gate did not pass. Per protocol, the run stopped before eight-region embedding and arbitrary-offset cropping. Coding, JPEG/noise, rotation, and confidence-filtering phases were not run and are reported as NOT RUN rather than as zero performance.

## A. What was successfully executed

- Created an isolated CPython 3.10.14 Conda environment.
- Cloned the official `facebookresearch/watermark-anything` repository and checked out commit `2c08af04d037d5667c02f6ddebbda9ff04581c3e` in detached-HEAD state.
- Installed PyTorch 2.5.1, Torchvision 0.20.1 and the complete pinned upstream requirements. `pip check` reported no broken requirements. Complete Conda and pip locks are saved.
- Downloaded the official MIT checkpoint from `https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth`.
- Loaded the official 94,416,643-parameter WAM model and ran inference on CPU without modifying the checkpoint or WAM source.
- Prepared ten public natural/scientific sample photographs bundled with scikit-image 0.24.0, recorded source and processed hashes, and used three fixed payload seeds per image.
- Ran 30 two-region and 30 four-region clean multi-message cases with official WAM embedding, detection, and README DBSCAN parameters.
- Saved every case result, environment details, image manifest, logs, and representative visualizations.

The ten test images were astronaut, coffee, Chelsea the cat, rocket launch, Hubble deep field, immunohistochemistry, retina, motorcycle, camera, and coins. The scikit-image data documentation records public-domain, CC0, or no-known-restriction status for its sample images; the exact installed source file hashes are in `official_results/image_manifest.json`.

## B. Environment and exact model versions

| Item | Executed value |
|---|---|
| OS / architecture | macOS 26.6.2 / arm64 |
| Python | 3.10.14 |
| Device | CPU; MPS unavailable to this process |
| PyTorch / Torchvision | 2.5.1 / 0.20.1 |
| NumPy / Pillow | 1.26.4 / 10.4.0 |
| WAM source | Official repository, commit `2c08af04d037d5667c02f6ddebbda9ff04581c3e` |
| Model configuration | Official `checkpoints/params.json`: 32 bits, `scaling_w=2.0`, `jnd_1_3_blue` |
| Checkpoint | Official `wam_mit.pth`, 377,825,938 bytes |
| Checkpoint SHA-256 | `90ef232384e023bd63245eb0c131abd69d2afc7b8f17a71ccedceb542bf009e2` |
| Detection | Official `multiwm_dbscan`, epsilon 1, min_samples 500, mask threshold 0.5 |
| Fixed seed | 20261007 |

No training, fine-tuning, alternate model, custom WAM approximation, original-image-assisted decoding, or coordinate-assisted decoding was used.

## C. Clean multi-region WAM result

The protocol was frozen before measurement. Every localized region occupied 110 x 205 pixels (8.60% of a 512 x 512 image). Two-message tests used opposite corners; four-message tests used all four corners.

| Metric | 2 regions | 4 regions |
|---|---:|---:|
| Image/payload cases | 30 | 30 |
| Exact packets | 57 / 60 (95.00%) | 112 / 120 (93.33%) |
| Images with all packets exact | 27 / 30 (90.00%) | 22 / 30 (73.33%) |
| Mean matched bit accuracy | 99.740% | 99.583% |
| Mean detected clusters | 2.00 | 4.00 |
| Mean localization union IoU | 0.9497 | 0.9443 |
| Mean PSNR | 47.15 dB | 44.08 dB |
| Mean RGB SSIM | 0.99374 | 0.98841 |
| Mean embedding time / image | 0.340 s | 0.679 s |
| Mean detect + cluster time / image | 0.558 s | 0.993 s |

The two-region gate passed exactly at its thresholds: 95% exact packet recall and 90% all-exact images.

The four-region gate required at least 90% exact packet recall and 80% all-four-exact images. Packet recall passed, but all-four-exact image recovery was only 73.33%, so the gate failed.

Every four-region image produced four DBSCAN clusters. The eight bad packets were decoding errors inside detected clusters: three had one wrong bit, two had two wrong bits, and three had three wrong bits. The main observed failure was therefore exact message decoding rather than missing localization or missing clusters. Bad clusters had a lower derived mean logit-margin confidence than exact clusters (0.879 versus 0.980), but distributions overlap and no threshold was calibrated on a separate development set.

## D. Arbitrary-offset crop result

**NOT RUN.** The clean four-region gate failed. In accordance with the protocol and primary stop condition, the experiment did not attempt eight simultaneous messages or arbitrary-offset crops. There is no executed evidence that this configuration can convert unknown cropping into recoverable packet erasures.

## E. MDS versus repetition versus RLNC

**NOT RUN with WAM.** The coding phase was conditional on the unknown-offset crop gate, which was not reached.

The previous synthetic coding-only pilot remains separate evidence: at exactly four retained packets, MDS recovered 70/70 subsets, systematic RLNC 69/70, and repetition 16/70. That supports choosing MDS over RLNC as the fixed 4-of-8 code if localized packet recovery is later validated. It does not establish end-to-end WAM crop robustness.

## F. JPEG / noise diagnostic

**NOT RUN.** There is no evidence from this sprint that JPEG or noise requires inner ECC. Adding ECC now would be premature, although the clean one-to-three-bit packet errors show that packet-level bit corruption exists even before those attacks.

## G. Rotation diagnostic

**NOT RUN.** No angle search was run because the earlier clean gate failed. No claim can be made about whether a test-time rotation search restores WAM recovery.

## H. Representative visualizations

- `official_results/visualizations/clean_2_regions.png`: original, masks, marked image, amplified difference, predicted localization, and message clusters.
- `official_results/visualizations/clean_4_regions.png`: the same six panels for four localized messages.

Visual inspection confirms that the regions are localized and separated. The aggregate tables, rather than the representative image, determine the decision.

## I. Failure cases

The four-region failures occurred in 8 of 30 cases: coffee seeds 1 and 2; Chelsea seed 1; rocket seed 2; Hubble seed 0; immunohistochemistry seeds 0 and 2; retina seed 2. Each returned four clusters, but one packet contained one to three wrong bits. All failures are retained in `official_results/clean_cases.csv`; none were dropped.

## J. Final questions

1. **Can WAM recover multiple different localized messages on clean natural images?** Yes in many cases, but not reliably enough for this exact-packet method: four-message all-exact recovery was 73.33%, below the frozen 80% gate.
2. **Can it still find and decode surviving messages after arbitrary-offset cropping without crop coordinates?** Unknown. The required clean gate failed, so crop testing was correctly stopped.
3. **Does erasure coding improve exact global payload recovery over a strong repetition baseline at comparable visual distortion?** Unknown for WAM; the end-to-end coding comparison was not reached.
4. **Is MDS preferable to RLNC for fixed 4-of-8?** At the coding layer, yes: the previous exhaustive pilot favored deterministic MDS (70/70) over the tested RLNC matrix (69/70). End-to-end WAM evidence is absent.
5. **Is JPEG a problem requiring inner ECC?** Unknown. JPEG was not tested after the clean gate failure.
6. **Can a tiny angle search meaningfully improve rotation robustness?** Unknown. It was not run.
7. **Is confidence-aware packet rejection useful?** Not established. The derived margin is lower on wrong packets, so it is a plausible future diagnostic, but it needs a separate development set and cannot repair insufficient packet count by itself.
8. **What should the final CV course method be?** Do not commit to localized WAM + MDS 4-of-8 yet. The current proposal is NO-GO. A future revised candidate would first need to improve clean exact packet reliability through a predeclared change such as larger regions, a calibrated embedding strength, or a compact inner code, and then repeat the same clean and blind-crop gates before any global erasure-code claim.

## Artifacts and reproducibility

- `PROTOCOL.md`: frozen gates and design.
- `RUNBOOK.md`: commands used for environment, source, checkpoint, and execution.
- `environment-conda-explicit.txt` and `environment-pip-lock.txt`: installed environment locks.
- `official_results/environment.json`: model, runtime, checkpoint, device, and seed.
- `official_results/image_manifest.json`: source and processed image hashes.
- `official_results/clean_cases.csv`: all 60 measured image/payload cases.
- `official_results/clean_summary.json` and `final_result.json`: machine-readable results and decision.
- `logs/main_validation.log`: complete main-run console output.

The main run completed in 82.35 seconds after model imports, using CPU. The earlier repository contents were verified unchanged against 35 stored SHA-256 values.

## Sources

- Official WAM repository and inference example: https://github.com/facebookresearch/watermark-anything
- Pinned source commit: https://github.com/facebookresearch/watermark-anything/tree/2c08af04d037d5667c02f6ddebbda9ff04581c3e
- WAM paper: https://arxiv.org/abs/2411.07231
- scikit-image sample-data documentation and provenance: https://scikit-image.org/docs/stable/api/skimage.data.html
