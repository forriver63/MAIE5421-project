# Four learning-based baselines: shared team protocol

This folder contains **HiDDeN, ReDMark, RoSteALS and SepMark** under the team's ten-image protocol. ReDMark/RoSteALS A/B are padding settings, not additional models. The original [100-image HiDDeN study](../README.md) remains unchanged; do not pool its family-macro scores with these flat fourteen-condition averages.

## Results

See [English report](REPORT_EN.html), [report insertion text](REPORT_INSERT_EN.txt), [overall.csv](overall.csv) and [conditions.csv](conditions.csv). Full307/420=73.10%; HiDDeN72/420=17.14%; ReDMark A/B135/420=32.14%; RoSteALS A214/420=50.95%, B211/420=50.24%; SepMark170/420=40.48%. Full has higher observed mean recovery and clean PSNR under these settings; it does not win every attack. JPEG/noise favor several baselines. Eight rotation conditions strongly influence the flat mean.

## Protocol and scoring

images.json records the ten public image names, order and hashes. Three32-bit payloads per image, seed20261008, fourteen identical attack conditions,420 cases per setting. protocol.py reproduces teammate attacks from learned-baseline commit2115087fe9d14f02dc70791d66e91addab1ddd6b. Use the shared source_for function; remaining padding comes from NumPy default_rng([20261008,image_index,seed,1]) (A uses zeros). Default transform512, RGB8 before attacks; PSNR/SSIM measure clean marked RGB images.

HiDDeN default0.45/native512 main score checks only32 useful bits, not its16-bit marker. hidden_scoring_sensitivity.csv retains marker-gated/other-strength views. ReDMark1024 information bits and native repetition; RoSteALS56 information bits plus native BCH; SepMark128 bits and robust Tracer decoder. Capacities and quality are not matched. See model setup: [HiDDeN](../hidden/README.md), [ReDMark](../redmark/README.md), [RoSteALS](../rosteals/README.md), [SepMark](../sepmark/README.md).

## Independent cached verification

cases.jsonl.gz contains2940 cases: Full plus six baseline/settings. Each retains decoder predictions, useful truth and outcome; no images or checkpoints. Run from repository root:

```sh
python baseline/team10/verify_scores.py
```

Requires NumPy and bchlib2.1.3. The verifier recomputes useful-message outcomes, validates deterministic messages and all98 per-condition rows. It verifies Full's reported payload against truth, not its neural localization. Adapters are reusable inference components requiring external author assets; this is not an automatic dependency/download installer. WM_ROOT is an external asset root, never a hard-coded machine path. assets.sha256.json records observed integrity, not an author cryptographic signature.

## Limits

No unmarked-image false-positive experiment was added: wrong-output rates here apply to marked images. Ten images, not420 cases, are the bootstrap units. RoSteALS rejects uncorrectable BCH packets; other baselines generally return a payload. SepMark's face-domain training differs from these natural images. Selection was not outcome-blind; results characterize these fixed settings rather than all published alternatives. Earlier experiments remain archived separately. No weights, source datasets or local process records are included.
