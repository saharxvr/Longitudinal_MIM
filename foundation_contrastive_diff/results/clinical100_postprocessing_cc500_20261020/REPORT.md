# Component cutoff comparison at fixed threshold 0.75

Exploratory comparison on the same repeatedly used 100 clinical cases and five
clinicians. Not independent test evidence. Cutoffs are sign-specific component
areas in equivalent 512x512 pixels, NOT numbers of connected components.
Retain components strictly larger than cutoff * native_area / 512^2.

| Cutoff | Original CE mean F1 | Augmented CE mean F1 |
|---|---:|---:|
| No filtering | 0.3332 | 0.3297 |
| 250 | 0.4376 | 0.4399 |
| 500 | 0.4613 | 0.4574 |
| 1000 | 0.4766 | 0.4884 |

All values use the SAME score threshold0.75 and mean combined detection F1 over
five clinicians. Historical Itamar100 mean is0.4646, including substitute maps
for pairs99/100. Thus cutoff500 is numerically close but below that reference;
cutoff250 is lower. Do not claim significance or independent superiority.

## Per clinician at fixed threshold

| Clinician | Original250 | Original500 | Augmented250 | Augmented500 |
|---|---:|---:|---:|---:|
| Avi | 0.4259 | 0.4554 | 0.4214 | 0.4438 |
| Benny | 0.3934 | 0.4118 | 0.3850 | 0.3922 |
| Sigal | 0.4424 | 0.4667 | 0.4611 | 0.4795 |
| Smadar | 0.4386 | 0.4611 | 0.4451 | 0.4675 |
| Nitzan | 0.4879 | 0.5114 | 0.4870 | 0.5041 |

The best500 thresholds across the broader exploratory sweep are0.75 for original
(F1 .4613) and0.9 for augmented (F1 .4747). These are not the matched-threshold
comparison above.

Mean predicted image area at cutoff250/500/1000 is13.34/13.31/13.25% for original,
13.99/13.97/13.88% for augmented. Large detection-F1 changes accompany very small
area changes: small fragments strongly affect detection counts. This does not
prove all removed regions are noise, nor establish safety for small findings.

The expanded sweep contains44settings/model and reproduces all30 published
baseline F1 values exactly. Three unit tests passed. Same protocol as the
[preceding sweep](../clinical100_postprocessing_sweep_20261020_verified/REPORT.md),
with cutoff500 added. Full results: [summary](summary.csv),
[clinicians](per_clinician.csv), [matching counts](per_pair_counts.csv.gz),
[protocol](protocol.json).
