# Four learned baselines on DIV2K

We compare Full with HiDDeN, ReDMark, RoSteALS and SepMark on DIV2K validation images 0801–0900. Each setting uses three 32-bit messages per image. The original Full and HiDDeN records supply the reference measurements. The three additional models use the same images, messages and attack generation. A uses zero padding and B uses seeded random padding.

The [20-condition table](core20_metrics.csv) gives exact recovery, wrong output and erasure counts and rates for 300 marked cases per row. PSNR and SSIM measure embedding quality before attacks. HiDDeN uses strength 0.45 with payload-only scoring. The [marker-gated supplement](hidden_gated_supplement.csv) scores the same outputs with marker rejection. The [summary](main_comparison_percent.csv) gives equal weight to six core attack families. The [per-condition results](attack_summary.csv) cover all 28 conditions including eight extended attacks. The [recovery figure](attack_recovery.png) shows results by condition. The compressed case table includes alternative HiDDeN operating points. Operating points differ in embedding quality and native capacity.

From the repository root, run `python baseline/div2k100/verify.py` to check the per-condition tables against the included records. This requires NumPy and pandas.

To run fresh inference for an added model, install its dependencies and prepare the upstream source and weights following the [ReDMark](../redmark/README.md), [RoSteALS](../rosteals/README.md) or [SepMark](../sepmark/README.md) instructions. Use a common asset root with the adapter's `third_party/` and `models/` paths and the processed test images under `processed/` as listed in the [image manifest](../image_manifest.json). The runner checks each processed image's SHA-256. ReDMark requires a TensorFlow GPU. The published GPU runs used PyTorch 2.5.1, NumPy 1.26.4 and TensorFlow 2.15.1 for ReDMark. Use a fresh output directory for each invocation.

```sh
python baseline/div2k100/run.py --model sepmark --root ./assets --out ./outputs/sepmark100 --device cuda
```

The same command accepts `redmark` and `rosteals`. Add `--limit 1` for a one-image run. The [HiDDeN study](../hidden/README.md) describes the original reference experiments.
