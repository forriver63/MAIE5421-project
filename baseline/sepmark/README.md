# SepMark Tracer

Wu, Liao and Ou, SepMark: Deep Separable Watermarking for Unified Source Tracing and Deepfake Detection, ACM MM2023. https://doi.org/10.1145/3581783.3612471 . Author code: https://github.com/sh1newu/SepMark . Non-commercial academic use; retain author notices.

Author-linked weights: https://drive.google.com/drive/folders/1h93NcAJXE21CsDluMyDdBdKGY5aV1pLC . Select Dual_watermark_256_128_0.1_0.0002_0.5_se_se_1_10_10_10_0.1_2023_04_18_16_29_54/models/EC_90.pth (file ID1k6Sap1XvjcT8gZpyJ78l_PWFV_6tEnF3). Do not use the separate30-bit model. Place source at WM_ROOT/third_party/sepmark and weight at WM_ROOT/models/sepmark/EC_90.pth. Hash is in ../team10/assets.sha256.json.

`adapter.py` exports `SepMark.embed(rgb_uint8, bits128)` and `SepMark.extract(received_rgb)`. The encoder and robust Tracer decoder (`decoder_C`) load 106 and 83 inference tensors, respectively, with strict matching. The model takes ±0.1 message values, uses embedding strength 1 and thresholds decoder logits at zero. The residual is resized from 256 to 512 pixels and decoding uses the attacked RGB8 image. Dependencies are the project PyTorch stack.

Both settings use the same first 32 payload bits. A fills the remaining 96 bits with zeros and B fills them with seeded random bits. Exact recovery of the first 32 bits is 172/420 (40.95%) for A and 170/420 (40.48%) for B. See the [shared protocol and paired comparison](../team10/README.md).

Three independent calibration images passed native128-bit and adapted32-bit clean recovery and same-received-image logit equivalence. This model was developed for faces; natural-image results are cross-domain and do not establish universal method ranking. See [shared protocol and results](../team10/README.md).
