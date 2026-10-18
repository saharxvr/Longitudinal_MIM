# GT Threshold Comparison

- Full-cohort measurements use saved simulated-float16 GT, not a new cache read.
- Rotation/reflection preserve pixel counts and 8-connected component topology.
- Exact examples are selected TRAIN cases, not representative cohort estimates.
- CCs are sign-separated, 8-connected; tiny means strictly fewer than 5 or 50 pixels.
- More components need not mean more changed area: a cutoff can split a component.
- All-background controls have undefined support/mass retention, not zero retention.
- Fewer pixels/components do not prove more biologically correct targets.
- No CC removal, GT normalization, model inference, training or test-split tuning.

[Full summary](summary.csv) | [Provenance and measurements](report.json)

## Full Train/Validation Retention

| Split | Cutoff | Active / originally active | Maps erased | Support retained | Absolute mass retained | Mean image area | Full-cohort CC available |
|---|---:|---:|---:|---:|---:|---:|---|
| train | 0.001 | 9508 / 9508 | 0 | 85.684% | 99.933% | 14.162% | True |
| train | 0.01 | 9466 / 9508 | 42 | 72.481% | 98.567% | 11.980% | True |
| train | 0.02 | 9426 / 9508 | 82 | 60.460% | 94.634% | 9.993% | True |
| val | 0.001 | 1293 / 1293 | 0 | 85.821% | 99.937% | 15.096% | True |
| val | 0.01 | 1284 / 1293 | 9 | 72.559% | 98.631% | 12.763% | True |
| val | 0.02 | 1279 / 1293 | 14 | 60.377% | 94.808% | 10.620% | True |
