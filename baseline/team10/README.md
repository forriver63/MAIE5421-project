# Four learning-based baselines: shared team protocol

This folder contains **HiDDeN, ReDMark, RoSteALS and SepMark** under the team's ten-image protocol. ReDMark, RoSteALS and SepMark each have A/B padding settings. The [100-image HiDDeN study](../README.md) uses a separate evaluation protocol.

## Results

Full recovered 73.10% of messages, HiDDeN 17.14%, ReDMark A/B 32.14%, RoSteALS A 50.95% and B 50.24%, and SepMark A 40.95% and B 40.48%. Several baselines led on JPEG and noise. Eight rotation conditions contribute to the flat mean. See the [English report](REPORT_EN.html), [report text](REPORT_INSERT_EN.txt), [overall results](overall.csv) and [per-condition results](conditions.csv).

SepMark A and B use identical payloads, images and attacks, with zeros or seeded random bits in the 96 padding positions. The difference in exact 32-bit recovery was +0.48 percentage points for A, with a paired image-bootstrap 95% interval of [−0.48, +1.67]. See the [padding comparison statistics](sepmark_padding_ci.json).

## Protocol and scoring

images.json records the ten public image names, order and hashes. Three32-bit payloads per image, seed20261008, fourteen identical attack conditions,420 cases per setting. protocol.py reproduces teammate attacks from learned-baseline commit2115087fe9d14f02dc70791d66e91addab1ddd6b. Use the shared source_for function; remaining padding comes from NumPy default_rng([20261008,image_index,seed,1]) (A uses zeros). Default transform512, RGB8 before attacks; PSNR/SSIM measure clean marked RGB images.

HiDDeN default0.45/native512 main score checks only32 useful bits, not its16-bit marker. hidden_scoring_sensitivity.csv retains marker-gated/other-strength views. ReDMark1024 information bits and native repetition; RoSteALS56 information bits plus native BCH; SepMark128 bits and robust Tracer decoder. Capacities and quality are not matched. See model setup: [HiDDeN](../hidden/README.md), [ReDMark](../redmark/README.md), [RoSteALS](../rosteals/README.md), [SepMark](../sepmark/README.md).

## Independent cached verification

`cases.jsonl.gz` contains 3,360 cases across Full and seven baseline settings. Each record retains the decoder predictions, expected payload and outcome. Run the verifier from the repository root.

```sh
python baseline/team10/verify_scores.py
```

Requires NumPy and bchlib 2.1.3. The verifier reconstructs deterministic payloads and padding, checks the full case grid and verifies all 112 per-condition rows and overall metrics. For Full it verifies the cached receiver output. The adapters require external author assets under `WM_ROOT`. The asset hashes record the files used in the experiment.

## Limits

No unmarked-image false-positive experiment was added: wrong-output rates here apply to marked images. Ten images, not420 cases, are the bootstrap units. RoSteALS rejects uncorrectable BCH packets; other baselines generally return a payload. SepMark's face-domain training differs from these natural images. Selection was not outcome-blind; results characterize these fixed settings rather than all published alternatives. Earlier experiments remain archived separately. No weights, source datasets or local process records are included.
