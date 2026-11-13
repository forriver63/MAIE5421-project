# SepMark Tracer

Wu, Liao and Ou, SepMark: Deep Separable Watermarking for Unified Source Tracing and Deepfake Detection, ACM MM2023. https://doi.org/10.1145/3581783.3612471 . Author code: https://github.com/sh1newu/SepMark . Non-commercial academic use; retain author notices.

Author-linked weights: https://drive.google.com/drive/folders/1h93NcAJXE21CsDluMyDdBdKGY5aV1pLC . Select Dual_watermark_256_128_0.1_0.0002_0.5_se_se_1_10_10_10_0.1_2023_04_18_16_29_54/models/EC_90.pth (file ID1k6Sap1XvjcT8gZpyJ78l_PWFV_6tEnF3). Do not use the separate30-bit model. Place source at WM_ROOT/third_party/sepmark and weight at WM_ROOT/models/sepmark/EC_90.pth. Hash is in ../team10/assets.sha256.json.

adapter.py exports SepMark: .embed(rgb_uint8,bits128), .extract(received_rgb). Encoder106 and robust decoder_C83 inference entries strictly load. The package bootstrap excludes an unused training-only loss import without changing network forward operations. Messages use +/-0.1, strength1, first32 useful bits plus96 seeded random bits. Decode threshold0. The semi-robust decoder_RF is not used. Native256 with teammate-style residual upsampling to512; receive the actual RGB8 attacked image. Dependencies are the project PyTorch stack.

Three independent calibration images passed native128-bit and adapted32-bit clean recovery and same-received-image logit equivalence. This model was developed for faces; natural-image results are cross-domain and do not establish universal method ranking. See [shared protocol and results](../team10/README.md).
