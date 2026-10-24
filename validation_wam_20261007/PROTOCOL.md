# Official WAM minimal validation protocol

> Archive navigation note (2026-10-22): this is a historical protocol/report. Script names and machine-local artifacts below describe the original run and may no longer exist in this checkout. See [current commands](../README.md), [migration coverage](../docs/RESEARCH_HISTORY.md), and [中文实验流程](../docs/WORKFLOW_ZH.md). Original measurements and protocol text below are retained.

Protocol frozen before the main measurement run on 2026-10-07. The earlier `quick_probe.py` is a clearly labeled engineering smoke test on one image and is not included in aggregate results.

## Fixed implementation

- Official repository: `facebookresearch/watermark-anything`
- Git commit: `2c08af04d037d5667c02f6ddebbda9ff04581c3e`
- Checkpoint: official `wam_mit.pth`
- Checkpoint SHA-256: `90ef232384e023bd63245eb0c131abd69d2afc7b8f17a71ccedceb542bf009e2`
- Official model configuration: `checkpoints/params.json`, including 32 message bits and `scaling_w=2.0`
- Official detector and `multiwm_dbscan`, with README parameters `epsilon=1`, `min_samples=500`, mask threshold 0.5
- No training, fine-tuning, checkpoint modification, replacement model, or detector access to the original image, region coordinates, crop offset, or crop rectangle

## Images and seeds

Ten public sample photographs bundled with scikit-image 0.24.0 are converted to RGB and center-cropped/resized to 512 x 512: astronaut, coffee, chelsea, rocket, Hubble deep field, immunohistochemistry, retina, motorcycle, camera, and coins. Source and processed SHA-256 hashes are recorded.

Clean experiments use three fixed payload seeds per image. Crop screening uses payload seed 0 from every image to keep the method-screening run bounded.

## Spatial regions

Eight fixed, non-overlapping rectangles form a 2 x 4 grid. Each is 110 x 205 pixels, or 8.60% of the 512 x 512 image. Two-region tests use opposite corners; four-region tests use all four corners; eight-region tests use the full grid. Region coordinates are used only for embedding and ground-truth scoring. Detection receives only the resulting image.

## Clean decision gate

Run two regions first. Continue to four only if exact packet recall is at least 95% and both-packets-exact image rate is at least 90%.

Continue to eight only if four-region exact packet recall is at least 90% and all-four-exact image rate is at least 80%.

Proceed to cropping only if eight-region exact packet recall is at least 85% and at least four exact packets are recovered in at least 90% of clean images. Any failed gate stops the later phases and yields NO-GO for the proposed 4-of-8 method.

## Unknown-offset crop gate

For each of the ten seed-0 eight-region images, test retained areas 75%, 50%, and 25%. At each level use six deterministic random offsets: two each at aspect ratios 1:1, 4:3, and 3:4. WAM runs directly on the cropped image. Crop coordinates are logged for scoring but never passed to WAM or the clustering decoder.

Proceed to erasure coding only if at least four exact messages are recovered in at least 60% of the combined 50%- and 75%-retained crops. This threshold is a screening rule, not a statistical claim.

## Coding gate, if reached

Compare raw/padded, strong interleaved repetition, systematic RLNC, and MDS at K=4, N=8. Every arm carries a 64-bit useful payload through eight 32-bit WAM messages. Coded packets use 8-bit packet ID, 16-bit symbol, and CRC-8. Coefficient tables are public protocol state and all overhead is reported. MDS must improve exact 64-bit recovery over repetition in the 50%- and 75%-retained crop set for GO.

## Diagnostics, if the crop gate passes

Run JPEG Q75/Q50, Gaussian noise sigma 2/255 and 5/255, and rotations +/-1, +/-2, +/-5, +/-10 degrees on MDS images. Only if small rotation causes failure, run one inverse-angle search with candidates -10, -5, -2, 0, 2, 5, and 10 degrees. Do not add inner ECC or a synchronization model in this sprint.

## Metrics

PSNR, RGB SSIM, detected clusters, union-mask precision/recall/IoU, one-to-one matched bit accuracy, exact packet recovery, false message clusters, derived cluster margin, exact 64-bit recovery when coding is reached, runtime, device, versions, attack parameters, and all failures.

## Final rule

GO requires all clean and crop gates plus a positive MDS exact-global-recovery improvement over strong repetition at comparable visual distortion. Otherwise the result is NO-GO for adopting the proposed method based on this sprint.
