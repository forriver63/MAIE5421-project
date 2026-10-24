# Frozen final hierarchy protocol — 2026-10-08

> Archive navigation note (2026-10-22): this is a historical protocol/report. Script names and machine-local artifacts below describe the original run and may no longer exist in this checkout. See [current commands](../README.md), [migration coverage](../docs/RESEARCH_HISTORY.md), and [中文实验流程](../docs/WORKFLOW_ZH.md). Original measurements and protocol text below are retained.

This protocol was written before any measurement in this directory. Reuse the official WAM environment, 10 hashed images, BCH(31,16)+parity codec, and exact prior four-region crop coordinates. Previous result directories are read-only.

## Packet codec

Use a non-reflected, MSB-first CRC-5 over the 11-bit word `packet_id[3] || symbol[8]`, with polynomial `x^5+x^2+1` (`0x25` including leading term), zero initialization, no final XOR, and five zero augmentation bits. The 5-bit remainder is appended to make `packet_id[3] || symbol[8] || crc[5]` exactly 16 bits. Encode those bits using the verified primitive BCH(31,16,7) plus overall parity to 32 WAM bits. BCH corrects up to three bit errors; parity is an explicit component of the three-error budget. CRC is checked only after BCH. CRC failure, BCH failure, or duplicate-ID conflict is an erasure. A CRC-valid wrong packet is a silent error. No confidence threshold is used.

Verify all 2,048 possible ID/symbol inputs with zero errors, exhaustive error patterns of Hamming weight 1–6 on the zero packet (the overall codec is linear), and random nonzero-message injections. Record reject, correct, and silent-wrong counts at each weight.

## Phase 1: same four-region blind crop protocol

Ten prior natural images × three payload seeds; four prior 110×205 corner masks; same 512×512 WAM settings. Each region gets its deterministic physical ID 0–3 and one deterministic 8-bit symbol. Embed once per image/seed and test the clean image plus the exact six prior crop variants per case: ~75% and ~50% image area, each with 1:1, 4:3 and 3:4 aspect ratios and the same deterministic random offsets from the previous sprint. The detector receives only marked pixels or cropped marked pixels. Oracle region masks and crop coordinates are applied after decoding solely for scoring.

Phase 1 passes only if, across all 720 crop-region cases, silent accepted wrong packets <=0.5%; among regions with >=75% physical retention, >=95% have exact ID and symbol; >=95% of failed packets are decoder-visible erasures; mean clean PSNR falls by no more than 0.5 dB and mean clean SSIM by no more than 0.005 relative to the previous protected four-region run (44.09249 dB, 0.98841234). These thresholds operationalize the user gate. Any failure stops the sprint.

## Phase 2: eight clean packets

Only after Phase 1 passes. Use the prior eight 110×205 nonoverlapping masks on the same 512×512 natural images: ten images × three seeds. Give each region its physical ID 0–7 and a deterministic random symbol. No MDS yet. The decoder uses raw WAM + BCH + CRC and receives only the marked image. Spatial oracle is used only to score errors. Advance only if >=99% of images have at least four exact valid packets, mean exact valid packets >=7, silent wrong packets <=0.5% of 240 region cases, mean PSNR >=40 dB, and mean SSIM >=0.97. No geometry adjustment is planned or silently substituted.

## Phase 3: outer MDS algebra

Only after Phase 2 passes. Use GF(256) primitive polynomial `0x11d`. Encode four 8-bit source symbols using an eight-row, four-column Vandermonde matrix with evaluation points 1–8. Verify that every one of the 70 four-row subsets is invertible and recovers many deterministic test payloads. Any failure stops the sprint.

## Phase 4: fair repetition

The repetition matrix uses physical region order `[A,B,C,D,D,C,B,A]`, so the two copies of each source byte are spatially separated across the 2×4 region grid. Both methods use the same global 32-bit payload, eight WAM messages, 3-bit physical packet IDs, CRC-5, BCH/parity, masks, strength, images, and crop windows. The packet ID indexes physical region and the public code matrix. A duplicate ID with conflicting accepted values is erased; repetition copies that disagree cause global failure. No ground truth resolves conflicts.

## Phase 5: paired blind crop comparison

Only after Phases 1–3 pass. Ten images × three payload seeds × six crop windows (three aspect ratios at ~75% and ~50% retained area) per method: 180 paired crops. The two marked images differ only in the eight source/coded byte values. Decoder sees only each cropped marked image. CRC-valid packet IDs and symbols feed the outer decoder. MDS requires at least four unique valid IDs and consistent full equations; repetition requires at least one valid copy of all four source bytes, with no disagreement. Exact 32-bit global payload is the primary endpoint.

GO only if MDS minus repetition exact-global recovery is at least 10 percentage points across the combined 75%/50% paired set, is nonnegative at each retained-area level, the method mean clean PSNR differs by at most 0.5 dB and SSIM by at most 0.005, and silent wrong packet acceptance stays <=0.5% in each method. Otherwise report MODIFY or NO-GO. All rates, counts, quality, runtime, and individual failures remain in CSV/JSON.
