# GPT-5.4 Precision Heatmap Repeatability Study

This directory contains a reproducible repeatability experiment for the GPT-5.4
precision semantic-difference heatmap annotator.

## Study design

- Model: GPT-5.4.
- Output mode: precision heatmap JSON.
- Sample: 12 fixed pairs selected with seed `20261001`.
- Fresh repetitions: five.
- Primary analysis: agreement among the five fresh repetitions.
- Secondary analysis: comparison with a frozen copy of the original GPT-5.4
  precision result.
- Blinding: each fresh run may inspect only its run-specific prompt and the 12
  snapshotted input images in this directory.
- Coordinate stress cases: pairs 99 and 100 are included intentionally because
  their images are non-square.

The exact sample, file hashes, prompt hashes, model identifier, run nonces,
repository revision, and runtime metadata are recorded in `protocol.json`.

## Cache mitigation and limitations

Each repetition is executed in a new isolated GPT-5.4 agent session. Each run
prompt contains a unique deterministic nonce that has no clinical meaning and
is explicitly marked as metadata. This makes the request bytes distinct and
mitigates prompt/result cache reuse.

The Copilot agent runtime does not expose provider-side cache controls, cache
hit telemetry, temperature, or generation seed. Therefore, this study can
verify independent-session repeatability with cache-busting requests, but it
cannot prove the internal provider cache was disabled. This limitation is
preserved in `protocol.json` and must be reported with the results.

## Directory contract

- `precision_heatmap_annotator.txt`: immutable base prompt snapshot.
- `inputs/pairN/prior_current.png`: immutable input snapshots.
- `baseline_original/responses/`: frozen original GPT-5.4 responses.
- `runs/run_01` through `runs/run_05`: independent prompts, manifests,
  responses, and rendered maps.
- `analysis/`: generated metrics, tables, collages, and summary.
- `protocol.json`: machine-readable design and provenance.

## Reproduction

From the repository root:

```powershell
& '.\.venv\Scripts\python.exe' `
  '.\python_files\Sahar_work\setup_llm_repeatability_study.py'
```

The setup command deterministically recreates the sample and provenance
manifests. LLM generation itself must then be run five times in fresh GPT-5.4
sessions according to each `run_manifest.json`. After generation:

```powershell
& '.\.venv\Scripts\python.exe' `
  '.\python_files\Sahar_work\analyze_llm_repeatability_study.py'
```

The analyzer validates response schemas, renders native-coordinate maps, and
recomputes all reported repeatability statistics and collages.

## Results

The completed results and interpretation are documented in
[`RESULTS.md`](RESULTS.md). Machine-readable aggregate metrics are in
`analysis/summary.json`; pair-level and run-pair tables are CSV files under
`analysis/`.

Metric conventions:

- Semantic signature: sorted `(pathology, change_type)` findings; rationales,
  confidence, and heatmap points are excluded.
- Spatial support: pixels with absolute rendered heatmap magnitude at least
  `0.10`.
- Positive and negative Dice are calculated independently.
- When both compared runs have no support for a sign, Dice is defined as
  `1.0`; presence-agreement metrics should therefore be reported alongside
  Dice.
- Signed-map correlation uses the full continuous rendered maps.
