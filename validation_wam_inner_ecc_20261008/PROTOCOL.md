# Frozen staged protocol — 2026-10-08

> Archive navigation note (2026-10-22): this is a historical protocol/report. Script names and machine-local artifacts below describe the original run and may no longer exist in this checkout. See [current commands](../README.md), [migration coverage](../docs/RESEARCH_HISTORY.md), and [中文实验流程](../docs/WORKFLOW_ZH.md). Original measurements and protocol text below are retained.

This protocol is written before WAM measurements. The previous official WAM validation directory remains read-only.

## Phase 1: codec

Construct primitive narrow-sense binary BCH(31,16,7) over GF(32) using field polynomial `0x25`. Generate the degree-15 polynomial from cyclotomic cosets of roots alpha^1 through alpha^6. Append one overall parity bit, making a 32-bit even-parity code with minimum distance 8. The decoder searches a complete syndrome table for BCH errors of weight at most 3. It explicitly repairs a parity-bit error only when the combined correction count is at most 3. A mismatch requiring four corrections is rejected. Four errors should be detected, not repaired; five or more are outside the guarantee and may silently miscorrect. Verify all 65,536 clean codewords, their exact minimum distance, all 1–4-bit error patterns on the zero codeword, and randomized payload/error trials before WAM.

## Phase 2: clean four regions

Reuse exactly the ten source/processed images and four mask rectangles from the previous official experiment, with 3 payload seeds per image. Use the same WAM source, checkpoint, 512-pixel images, strength, detector, DBSCAN epsilon 1 and min_samples 500. Each of the four regions carries a deterministic random 16-bit useful payload protected to 32 bits. Embed with official WAM. Decode every cluster's raw 32-bit output with the codec, without truth or mask information. Match clusters to source regions **only for scoring**, using spatial overlap and a one-to-one assignment. Record missing and unmatched clusters explicitly. Compare the 30 paired image/seed cases with the prior uncoded run; payload bits differ because capacity is 16 versus 32 bits per region.

Decision gate: advance only if useful-payload packet recovery >=98%, all-four-exact images >=90%, silent wrong useful decodes <=1% of 120 packets, and mean PSNR falls by no more than 0.5 dB and mean SSIM by no more than 0.005 relative to the prior four-region means (44.0838 dB, 0.988414). All four conditions are required. This is a screening gate, not a formal equivalence or significance test.

## Phase 3, only after Phase 2 passes

Use the same four-message marked images. Test three deterministic random offsets at each of 75% and 50% retained area per image/seed, with aspect ratios 1:1, 4:3 and 3:4: 180 crops total. The decoder sees cropped pixels only. For scoring, project original masks into the crop and record physical retention per region. Define largely surviving as >=75% of original region area retained. Define a useful local channel as: at least 75% of largely surviving regions recover the exact useful payload at each crop level; at least one cluster is localized in >=80% of crops at each level; silent wrong decodes <=1% of all original regions; and physical removal accounts for the majority of missing packets. These screening thresholds are fixed now to operationalize the qualitative user gate. Stop if any fail.

## Phase 4, only after Phase 3 passes

Eight masks use the prior 2x4 grid (same 110x205 pixels) at 512x512 first, 5 images x 3 payload seeds. This is the smallest unchanged-geometry test. If not practical, a separately labeled larger-image diagnostic may be considered; it does not override a 512 failure. Record P(valid useful packets >=4), mean PSNR/SSIM, and cluster counts. Proceed only if >=90% of clean images recover at least four valid packets, mean PSNR >=40 dB, mean SSIM >=0.97, and localization yields at least four clusters in >=90% of images.

## Phase 5, only after Phase 4 passes

Use a 32-bit global payload, MDS(8,4) over GF(256), and a strong repetition arm under eight identical WAM regions, 16 useful bits per inner codeword. Metadata and integrity overhead must fit the 16 bits. Compare paired image, payload and unknown-offset crop conditions, with no oracle data at the decoder. Report exact global recovery and quality. The outer comparison must not run unless earlier gates pass.

Final decision is exactly one of GO, MODIFY, NO-GO. A failed clean or crop gate yields MODIFY when a specific local change remains plausible; NO-GO is reserved for evidence that rejects localized hierarchical coding for this project.
