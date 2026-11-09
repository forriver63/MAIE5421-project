# HiDDeN Meta JND 48-bit

Place these two files under the asset root's `models/hidden/` directory:

| File | Source | SHA-256 |
|---|---|---|
| encoder_with_jnd_48b.torchscript.pt | https://dl.fbaipublicfiles.com/ssl_watermarking/encoder_with_jnd_48b.torchscript.pt | 4722388872e8ec7cbc3084806546d852b982cc1ed01245d8dc039dd748af3071 |
| decoder_48b.torchscript.pt | https://dl.fbaipublicfiles.com/ssl_watermarking/decoder_48b.torchscript.pt | d6f0c052efb0a26869322ee657d9982bba79b32c2e0305fd4339aff45b58ac49 |

RGB pixels are ImageNet-normalized, message bits use -1/+1, and decoder logits use a zero threshold. The first 16 bits are marker 0xA5C3, followed by the 32 useful bits, MSB first. At most one marker error is accepted. No truth-dependent selection or extra ECC is used.

`native512` follows the archived OmnisealBench direct-resolution generator/detector. `resize256` is a sensitivity variant inspired by VideoSeal's resizing wrapper: infer at 256, convert the residual to pixel space using ImageNet standard deviations, resize the residual to the original shape, add to original pixels, clamp and quantize. The encoder's JND attenuation remains enabled. VideoSeal's builder uses different checkpoint names and scaling, so this variant is not an exact reproduction of the complete VideoSeal evaluation.

Before observing smoke results, the expansion rule was fixed: at strength 0.45, clean raw BER below 1%, or JPEG75 raw BER improvement of at least 2 percentage points. The 20-image calibration smoke did not meet either condition: clean BER 2.19% → 8.75%; JPEG75 12.66% → 12.19%. Therefore no 8,400-case resize256 test was selected. This limited check does not prove equivalence of all resolution protocols or all checkpoints.

The original strength 0.30, default 0.45 and calibration PSNR-nearest 0.3615234375 formal results are all retained. Closer PSNR does not mean matched SSIM. See `results/` and [the shared instructions](../README.md).
