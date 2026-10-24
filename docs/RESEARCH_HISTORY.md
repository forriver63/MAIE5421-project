# Research evidence, migration and audit

## Provenance and evidence policy

The original research repository is [forriver63/MAIE5421-project](https://github.com/forriver63/MAIE5421-project). The local source baseline is commit [`90a5738258c8df19366386643ca8cc4712b4b6cf`](https://github.com/forriver63/MAIE5421-project/tree/90a5738258c8df19366386643ca8cc4712b4b6cf). Deleted historical Python scripts can be inspected at that revision; the current package is a refactor with some reduced experimental coverage, not a complete historical workflow replay.

Four `validation_*` directories retain the original protocols, reports and selected measurements. They should be treated as evidence archives, not destinations for new runs. The working tree already contained changes to several historical result files and figures before the documentation audit; therefore it would be inaccurate to certify all local archived files as byte-identical or unchanged. Use the pinned original commit to inspect original evidence. This audit did not restore or overwrite those pre-existing edits.

See [WORKFLOW_ZH.md](WORKFLOW_ZH.md) for detailed experimental reasoning, sample counts, failure modes, command order and interpretation.

## Four research decisions

| Historical directory | Question and result | Why retained |
|---|---|---|
| `validation_wam_20261007/` | Original four-region raw recovery 112/120; all-four rate 22/30 failed the gate; localized failures had 1–3 bit errors | Motivation for inner ECC; results use `official_results/` |
| `validation_wam_inner_ecc_20261008/` | Protected clean run: 115/120 exact raw codewords, 120/120 payloads after BCH; crops had 8/720 silent errors, failing the 1% gate | BCH benefit and evidence that BCH alone is insufficient |
| `validation_wam_hierarchical_final_20261008/` | CRC crop screening, eight-packet feasibility, MDS algebra, paired crops: MDS 162/180 vs repetition 161/180 | Integrity filtering and a negative result for MDS selection |
| `validation_wam_final_robustness_20261008/` | Frozen repetition pipeline; 420 attacked/clean cases and ten unmarked controls; LIMITED | Final measured strengths and limitations |

The 115/120 count is before BCH on the protected run, not the original uncoded baseline. Protocol gates are screening decisions, not statistical significance claims. The same ten images are reused across stages, not an independent held-out benchmark.

## Current migration coverage

| Original implementation | Current entry point | Coverage / gap |
|---|---|---|
| `run_validation.py` | `experiments/baseline/run.py` | Clean 2/4/8 screening only; original downstream crop/coding/diagnostic phases are not replayed. `CLEAN_PASS` means only this stage passed. |
| `bch31.py`, `packet_codec.py`, `outer_codes.py` | `src/watermarking/codec/`, `tests/` | Shared codecs and exhaustive checks; pytest output is not the old verification JSON format. |
| `run_sprint.py` | `experiments/inner_ecc/run.py` | Four-region protected clean experiment; evaluates quality against the frozen original baseline constants. |
| `run_crop.py` | No current standalone equivalent | Historical BCH-only blind-crop evidence retained. |
| `run_phase1.py`, `run_phase2.py` | Shared inference/scoring in `src/watermarking/eval.py` | Four-region CRC crop gate and independent eight-packet feasibility runners are not migrated. |
| `run_final_comparison.py` | `experiments/hierarchical/run.py` | Paired outer comparison; gain, per-area, quality and conservative accepted-error gates. Does not rerun or enforce historical phase 1/2 prerequisites. |
| `run_robustness.py`, `finalize.py` | `experiments/robustness/` | Main 14 conditions, negatives, summaries/figure; inverse-angle search unimplemented and now fails explicitly if triggered. |
| `quick_probe.py`, `phase0_probe.py` | `wm embed` / `wm detect` for a manual smoke check | Not identical to the old engineering/environment probe. |
| `verify_artifacts.py`, old plotting/finalization helpers | No full equivalent | Current finalizer checks counts, but does not reproduce the old complete provenance/artifact audit. |
| No original component-removal study | `experiments/ablation/run.py` | Extra repetition/MDS comparison on clean and JPEG75, not BCH/CRC removal ablation. |

Current scripts are independent runners. Their outputs must not be described as proof that all the original sequential gates were rerun. Numerical equivalence of an entire experiment requires a separate measured comparison; code movement or unit tests alone are insufficient.

## Repairs in the 2026-10-22 documentation/code audit

- Updated setup, CLI/API commands, fresh-run and reuse instructions, output locations, research explanations and archive notes.
- Fixed non-512 image embedding by explicitly applying the same RGB center-fit geometry used by the benchmark. Detection keeps incoming dimensions.
- Resolved `WAM_PATH` before the model loader temporarily changes directories, allowing relative checkout paths.
- Added a scikit-image fetcher/cache fallback when a source sample is not bundled; the dependency is pinned to 0.24.0 and the fallback uses its internal `_fetch` API.
- Restored the missing eight-region baseline gate and stopped calling a clean-only pass an overall research GO.
- Replaced hard-coded BCH quality gate passes with measured deltas and restored the all-four recovery criterion.
- Restored per-area gain, quality, and conservative silent/unmatched-accepted error criteria to the outer comparison.
- Corrected robustness physical-retention measurements for cropped/rotated masks instead of reporting every region as fully retained.
- Added explicit robustness `--reuse` and `--output-dir`, checked all three CSV row counts before reuse, and avoided model/data loading for reuse. This is not a content/provenance audit or partial-run resume.
- Made a triggered but unimplemented inverse-angle search explicit instead of silently reporting experiment completion.
- Ensured the finalizer creates its figure directory and accepts the same output directory.
- Redirected upstream loader diagnostics to stderr during CLI detection so `wm detect --json` emits parseable JSON; bounded the web payload form to an unsigned 32-bit integer.
- Rejected unknown outer method names instead of silently treating every unknown method as MDS; clarified the actual ablation scope.

These repairs do not replace archived metrics with newly measured values. Existing `experiments/*/results/` may predate the repairs and need a fresh run to acquire new fields and measurements.

## Exclusions and future work

Do not commit environments, nested WAM Git history, model weights, caches, raw event logs or generated copies of the ten input images. Historical reports can mention files that are intentionally absent; their references are descriptions of past runs rather than current instructions.

Remaining work for a complete historical replay is explicit: migrate the BCH-only crop runner and CRC/eight-region prerequisite runners, restore full artifact/provenance checks, and implement the conditional inverse-angle search if required by future results. A true component ablation requires separately defined matched-budget controls and a protocol; merely naming the current script `ablation` does not supply that evidence.

## Audit verification

The eight existing codec tests and eight workflow regression tests passed in the local environment. Regression cases cover gate failures, image geometry, relative WAM paths, unknown methods, JSON output isolation, crop retention and dataset-cache fallback. A real checkpoint smoke test embedded `0x12345678`, saved a PNG, reopened it and recovered decimal `305419896`. Isolated copies of the existing robustness CSVs passed explicit reuse, wrapper execution, summary/figure generation and overwrite refusal. Python compilation, shell syntax and new documentation links were checked.

The complete 420-case inference benchmark and all historical stages were not rerun during this audit. Dependency deprecation warnings remain. A repository-wide whitespace check also reports CRLF lines in pre-existing modified historical CSVs; those numerical artifacts were left intact. Checks on the edited documentation and shell files passed.
