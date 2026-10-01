# GPT-5.4 Precision Repeatability Results

## Conclusion

GPT-5.4 was **not deterministic** in this experiment. Five fresh isolated
sessions frequently agreed that some change was present, but often disagreed
on the direction, pathology, and precise localization of that change.

These results describe repeatability under the Copilot agent runtime. They do
not establish the behavior of every GPT-5.4 endpoint or decoding configuration.

## Primary results: five fresh runs

The primary analysis contains 12 pairs and all 10 possible run pairs per image,
for 120 fresh-run comparisons.

| Measure | Agreement |
|---|---:|
| Exact JSON | 4.2% |
| Semantic signature | 20.8% |
| Any-change presence | 81.7% |
| Positive/worsening presence | 60.0% |
| Negative/improving presence | 58.3% |
| Pathology-set Jaccard | 0.388 |
| Positive support Dice | 0.455 |
| Negative support Dice | 0.514 |
| Continuous signed-map correlation | 0.202 |

Only **1 of 12 pairs** had the same semantic signature in all five fresh runs.
No pair had exactly identical JSON in all five runs. The mean number of unique
semantic signatures per pair was **3.67 out of five runs**.

Exact JSON agreement is not the main clinical stability measure because
rationale wording and coordinates can differ without changing the semantic
conclusion. However, the low semantic-signature agreement and signed-map
correlation show that variability was not merely textual.

## Pair-level observations

- Most stable: pair 92, with one semantic signature across all five runs and
  signed-map correlation `0.573`.
- Pair 99 had stable positive/negative presence but four distinct semantic
  signatures, demonstrating that stable direction does not imply stable
  pathology labeling.
- Pairs 47, 77, and 100 produced five distinct semantic signatures from five
  runs.
- Pairs 42, 47, and 81 had mean signed-map correlations at or below zero,
  indicating strongly inconsistent signed localization.
- Change/no-change presence was unstable for pairs 31, 51, and 68.

The full table is `analysis/per_pair_summary.csv`. Visual comparisons are in
`analysis/collages/`, with every overlay normalized independently.

## Secondary comparison with the original GPT-5.4 result

The frozen original result was excluded from the primary analysis. Across its
60 comparisons with fresh runs:

| Measure | Agreement |
|---|---:|
| Exact JSON | 0.0% |
| Semantic signature | 11.7% |
| Any-change presence | 78.3% |
| Positive/worsening presence | 51.7% |
| Negative/improving presence | 63.3% |
| Pathology-set Jaccard | 0.300 |
| Continuous signed-map correlation | 0.143 |

## Validation

- 60/60 fresh responses were present and schema-valid.
- 12/12 frozen baseline responses were schema-valid.
- 72/72 rendered NIfTI maps round-tripped with zero numerical error.
- All prompt, image, and baseline snapshot hashes matched `protocol.json`.
- All 12 repeatability collages were generated.
- Non-square pairs 99 and 100 rendered successfully.

## Cache-control limitation

Every fresh repetition used:

1. A new isolated GPT-5.4 session.
2. No access to baseline or other-run outputs.
3. A unique deterministic non-semantic nonce in the prompt.

This makes request contents distinct and mitigates cache reuse. The runtime does
not expose provider-side cache settings or cache-hit telemetry, so it is not
possible to prove that internal caching was disabled. Temperature and generation
seed are also not exposed. Claims should therefore use the phrase
**independent cache-busted sessions**, not **provider cache disabled**.

## Reproducible artifacts

- `protocol.json`: design, model, sample, hashes, nonces, and environment.
- `analysis/summary.json`: aggregate results.
- `analysis/response_manifest.csv`: response hashes and render validation.
- `analysis/pairwise_fresh_runs.csv`: all 120 primary comparisons.
- `analysis/original_baseline_vs_fresh.csv`: all 60 secondary comparisons.
- `analysis/per_pair_summary.csv`: pair-level stability.
- `analysis/collages/`: visual run-to-run comparisons.
