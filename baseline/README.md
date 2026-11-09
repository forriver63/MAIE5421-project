# Unified system comparison

This benchmark compares complete 32-bit message recovery under identical images, messages and attacks. Systems have different logical embedding budgets: Full uses eight 32-bit regional carriers (256 bits), HiDDeN uses 48 bits (16 marker + 32 payload). These are system comparisons; they do not isolate neural backbone quality or coding gains.

Read [REPORT_EN.html](REPORT_EN.html) for the current results and limitations, [TEAM_REPORT_INSERT_EN.txt](TEAM_REPORT_INSERT_EN.txt) for a compact report section, and [../ablation/README.md](../ablation/README.md) for component experiments.

## Setup and inputs

Use Python 3.10 and the project's pinned dependencies. The published benchmark used CPU PyTorch 2.5.1, not model training. Set `WM_ROOT` to an absolute local asset directory and `WAM_PATH` to its `third_party/watermark-anything` checkout. Choose storage locations appropriate for your machine. The author's Windows launch script separately enforces the local storage policy; the shared library is platform-independent.

1. Install this repository with `python -m pip install -e '.[dev]'` inside your environment. No administrator or user-site install is needed.
2. Clone `https://github.com/facebookresearch/watermark-anything.git` under `$WM_ROOT/third_party/watermark-anything`, and check out `2c08af04d037d5667c02f6ddebbda9ff04581c3e`. Set `WAM_PATH` accordingly. Obtain `checkpoints/wam_mit.pth` from `https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth`; retain upstream `checkpoints/params.json`. Expected weight SHA-256: `90ef232384e023bd63245eb0c131abd69d2afc7b8f17a71ccedceb542bf009e2`.
3. Put the official DIV2K `DIV2K_train_HR.zip` and `DIV2K_valid_HR.zip` in `$WM_ROOT/downloads`. Dataset source: https://data.vision.ee.ethz.ch/cvl/DIV2K/. Preparation extracts only training images 0001–0020 and validation images 0801–0900. Existing inputs need not be downloaded again.
4. Prepare deterministic RGB 512-square center crops:

```sh
python -c "import os; from watermarking.bench.manifest import prepare; prepare(os.environ['WM_ROOT'])"
```

`image_manifest.json` provides the expected identifiers and source/processed hashes. Its relative paths are relative to the asset `data/` directory. Compare the prepared manifest with these hashes; do not substitute the training images into the test split.

5. Copy `configs/protocol.example.json` to `configs/protocol.local.json`. `WM_ROOT` overrides `root`; otherwise `root` is relative to the config file. `manifest` is relative to that root. Choose a fresh output directory for each configuration; an identical fingerprint resumes automatically. `--resume` is an explicit reminder, not a way to bypass identity checks.
6. Obtain HiDDeN weights as described in [hidden/README.md](hidden/README.md), then run `python -m pytest`.

## Experiments

```sh
python baseline/hidden/run.py --config configs/protocol.local.json --run-dir output/hidden-psnr
python baseline/hidden/run.py --config configs/protocol.local.json --run-dir output/hidden-psnr --negative
```

The example strength is the calibration-selected closer-PSNR value 0.3615234375. Use separate configs/directories for 0.30 and the model-card default 0.45. Set `hidden_protocol` to `native512` for published formal runs.

For the resolution sensitivity and its calibration-only expansion decision, use the same prepared assets:

```sh
python baseline/protocol_study.py --root "$WM_ROOT" --out output/calibration
python baseline/review_calibration.py --root "$WM_ROOT" --out output/calibration --config configs/protocol.local.json
```

Calibration freezes before test inference. A full run has 100 images × 3 payloads × 28 attack conditions = 8,400 positives and 100 clean negatives. All attacks share `src/watermarking/bench/attack_registry.py`. Crop dimensions are passed directly to the receiver; neither the true payload nor crop origin is supplied. Embeddings undergo a lossless PNG round trip before attacks. Optional `--max-cases 512` bounds worker memory; rerun the identical command to resume.

Aggregate complete system runs and recompute paired intervals without model inference:

```sh
python baseline/analyze.py --run full=output/components/full --run hidden=output/hidden-psnr/hidden --out output/comparison
```

Each supplied directory must contain all 8,400 positives and 100 negatives. For decoding ablations use the finalizer in the ablation README, which retains each layout rather than treating layouts as independent images.

## Interpretation and attribution

Exact/wrong/erasure are mutually exclusive; Marker rejection or incomplete packet reconstruction counts as erasure. Raw BER ignores acceptance. Carrier BER for Full/E1 is conditional on localized regions and has a different denominator, so it cannot be ranked against global-message BER.

HiDDeN: Zhu et al., *HiDDeN: Hiding Data With Deep Networks*, ECCV 2018. We use Meta's later JND implementation, not the original authors' 2018 weights. References: https://github.com/facebookresearch/omnisealbench and https://github.com/facebookresearch/stable_signature. Preserve upstream attribution and noncommercial restrictions; no weight redistribution is included here. WAM source and checkpoint retain their upstream terms. DIV2K images are not included.

Additional teammate baselines should use this image manifest, payload generator, PNG quantization, attack plans and scoring definitions. Record native/effective payload budgets, detection rules, quality and calibration choices; shared attack names alone do not make independently produced scores comparable.

## Regenerate the published presentation

Run `python baseline/render_report.py` from the repository root to regenerate the English report and figures from the included summary CSV files. It does not rerun inference. `TEAM_REPORT_INSERT_EN.txt` is the concise section for integration into the team report; the HTML also includes detailed supplementary tables.

The example config reproduces the selected formal working point; the historical strength search target was Full calibration PSNR 39.18387246 dB on images 0001–0020. The independent calibration check and all three formal HiDDeN operating points are retained. No test images were used to select strength.
