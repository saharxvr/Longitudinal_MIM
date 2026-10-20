# Exploratory clinical100 postprocessing sweep

Tested 39 settings on each of two saved models (78 model/settings total), using
all 100 pairs and all five clinicians. No retraining or source-prediction changes.
These cases have been repeatedly reused. Settings below were selected on the
same cases: these are exploratory tuning results, NOT independent test gains.

## Validation and protocol

- Every input signed-map hash verified against its published verification file.
- Every annotation hash verified, allowing only Windows CRLF-to-LF conversion
  when required; affected files are explicitly listed in `protocol.json`.
- Unmodified threshold0.5 exactly reproduced all 30 published values:
  five clinicians x three directional/combined F1 values x two models.
- Optimized matcher passed 102 parity checks against the actual legacy matcher.
  Matching is reference-first greedy one-to-one, any same-sign overlap,
  8-connected components, no empty-empty boost.
- Three unit tests passed, covering matcher parity, sign-specific area filtering,
  nonmutation and legacy high-score suppression.
- Legacy annotation rasterization emitted existing degenerate-ellipse warnings.
  Baseline reproduction passed despite these warnings.

## Mean clinician detection F1

Each row except baseline reports its best threshold among the tested settings.
Threshold selection is by unweighted mean combined F1 over five clinicians,
not a separate optimal setting per clinician.

| Method | Original CE threshold | Original F1 | Augmented CE threshold | Augmented F1 |
|---|---:|---:|---:|---:|
| Baseline, no cleanup | 0.50 | 0.3141 | 0.50 | 0.3211 |
| Threshold only | 0.90 | 0.3454 | 0.90 | 0.3306 |
| Remove components <=50 equivalent pixels | 0.90 | 0.4019 | 0.90 | 0.4014 |
| Remove components <=250 equivalent pixels | 0.75 | 0.4376 | 0.90 | 0.4537 |
| Remove components <=1000 equivalent pixels | 0.75 | **0.4766** | 0.75 | **0.4884** |
| Gaussian sigma1 | 0.90 | 0.3695 | 0.90 | 0.3651 |
| Gaussian sigma2 | 0.90 | 0.3876 | 0.90 | 0.3873 |
| Gaussian sigma1 + remove <=250 | 0.90 | 0.4386 | 0.90 | 0.4512 |
| Literal Itamar function on native scores | 0.00 | 0.2650 | 0.00 | 0.2603 |

Modern component filters treat signs separately. A cutoff K means retain
components with native pixel count strictly greater than K * native_area / 512^2.
Thus 1000 equivalent pixels is about 0.3815% of the whole image.
Gaussian sigma is scaled per native axis relative to a 512-pixel image;
scores are smoothed before thresholding. Thresholds use strict inequalities.

The literal Itamar function was loaded directly from the legacy source without
importing its unrelated model dependencies. It retains its literal 250-native-
pixel component rule and 512^2 intensity-band rule, joins nonzero signs for that
component step, suppresses abs(score)>0.75, and uses asymmetric weak-score
suppression. It was applied to restored native maps, NOT the unavailable
historical pre-restoration model outputs. This tests the literal cleanup function
on our outputs, not an exact reproduction of Itamar's full inference pipeline.

## Per-clinician combined F1 at threshold0.75 + component cutoff1000

| Clinician | Original CE | Augmented CE |
|---|---:|---:|
| Avi | 0.4752 | 0.4739 |
| Benny | 0.4326 | 0.4259 |
| Sigal | 0.4875 | 0.5141 |
| Smadar | 0.4800 | 0.4984 |
| Nitzan | 0.5076 | 0.5298 |
| Mean | 0.4766 | 0.4884 |

Mean positive/negative F1 at this setting:

- Original: 0.4935 / 0.4553.
- Augmented: 0.4673 / 0.5137.

## Interpretation and limitations

At unchanged threshold0.5, component cutoff1000 alone increases mean F1 to
0.4718 / 0.4803. Much of the detection-score improvement therefore comes from
component filtering, not threshold changes alone. This is consistent with small
unmatched fragments affecting the detection metric, but is not proof that every
removed component is noise; real small changes may also be discarded.

At the best tested setting, mean predicted image area is still 13.25% / 13.88%.
On 12 empty-consensus cases, mean predicted area is 8.01% / 8.37%, and the worst
is 40.47% / 46.41%. Empty consensus is not proof of absence of clinical change.
Thus the improved detection F1 does not establish accurate boundaries or solved
false positives. Any-overlap matching can reward broad components.

The largest tested area cutoff won. This is a boundary result, not evidence of
the globally optimal cutoff; do not silently expand the search and present its
maximum as independent validation. Freeze a justified policy and test on new
data, or use synthetic validation to choose the policy for future runs.
No statistical significance claim is made.

## Files and reproduction

- [All model/settings averages](summary.csv)
- [All clinician-level F1 values](per_clinician.csv)
- [Per-pair matching counts](per_pair_counts.csv.gz)
- [Protocol and legacy source hash](protocol.json)

From the study directory:

```bash
python -m evaluation.sweep_clinical_postprocessing \
  --models results/clinical100_upright_ce_dice_thr0p001_20261019_091535 \
           results/clinical100_upright_ce_dice_thr0p001_aug_b16_20261020_131848 \
  --out-dir results/clinical100_postprocessing_NEW
```
