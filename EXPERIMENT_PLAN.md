# Historical pilot protocol (written before measurement)

> Archive note: this document describes the early 64-bit DWT/DCT pilot, not the final 32-bit WAM system. Current commands and the staged WAM experiment are documented in [README.md](README.md) and [docs/WORKFLOW_ZH.md](docs/WORKFLOW_ZH.md). The original protocol below is retained as historical context.

Scope: validate the latest referenced conversation's payload-coding hypothesis on actual images using the unmodified invisible-watermark dwtDct backend. WAM is a future transfer experiment, not a claimed result.

- 64 useful payload bits; K=4 source symbols of two bytes each.
- N=8 spatial packets; each 32 bits = 8-bit packet index + 16-bit symbol + CRC-8 (poly 0x07, init 0).
- Public, fixed coefficient table generated independently of the test payloads. GF(256), primitive polynomial 0x11d. First four RLNC rows systematic, four uniformly pseudorandom rows.
- Same eight 256x256 tiles on a 2x4 grid; images resized to 1024x512 for this pilot. Same default backend strength and gross budget of 256 embedded bits.
- Raw padded reference, spatially interleaved repetition, systematic RLNC; additionally an MDS Vandermonde erasure-code control to avoid claiming RLNC-specific novelty.
- Strong repetition layout: [0,1,2,3;2,3,0,1], two copies of each symbol. This layout covers all four symbols in every aligned rectangular crop containing at least four tiles. Pure aligned cropping alone need not favor RLNC.
- Ten public example images, three seeded independent 64-bit payloads per image (30 paired cases). Image is the independent sampling unit.
- Clean, JPEG Q75 and Q50, Gaussian noise sigma=2 and 5 (8-bit units), resize 0.75 then restore, and 2-degree rotation without registration.
- Grid-aligned contiguous crops retaining 4 and 6 tiles; paired crop+JPEG. Arbitrary-offset crops retaining 75% area use known crop coordinates and discard partial tiles. This is an explicitly registration-assisted upper-bound experiment.
- Coding-only exhaustive enumeration of all subsets of m packets, kept separate from image measurements.
- Main metric: exact recovery of all 64 useful bits, counting erasures as failure. Secondary: encoded packet BER, CRC packet acceptance, wrong accepted packet count, PSNR and RGB SSIM. No expected-payload oracle in decoder; ground truth only scores outputs.
- Report paired mean differences and 95% bootstrap confidence intervals clustered by original image (2,000 resamples).
- Decision: support continued development only if an attacked-image exact-recovery gain over strong repetition is positive with bootstrap lower bound >0, while mean PSNR differs by <=0.5 dB and mean SSIM by <=0.005. This pilot does not validate blind crop registration, neural transfer, security, or general benchmark superiority. Report every attack including failures.

## Environment amendment (before first measurement)
Package registry DNS and direct connection both timed out. No natural-image dataset or official DWT-DCT/WAM runtime is installed. The executed pilot therefore uses an explicitly separate NumPy implementation of Haar DWT + single DCT-coefficient QIM (step 80, 8x8 LL blocks, 8 repetitions per packet bit) and ten deterministic procedural synthetic fixtures. This is a mechanism/smoke test, not a natural-image benchmark and not a reproduction of invisible-watermark. The original official-backend runner remains available for follow-up. Conclusions must be provisional; even a positive result cannot establish WAM transfer or production robustness. Bootstrap intervals describe sensitivity across these fixtures only, not a sampled natural-image population. No embedding strength tuning after results.
