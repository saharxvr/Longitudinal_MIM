# Continuous GT Audit

Selected pairs: 18572; issues: 0.

[Summary CSV](summary.csv) | [Per-pair CSV](per_pair.csv) | [Report and histograms](report.json)

Raw means continuous transformed GT, not a sign mask. Scales on the raw-GT zoom/overlay
depend on GT alone, never prediction. Red=positive, blue=negative. Exact sign support
can include tiny numerical/interpolation tails; this report does not select a loss.

The CSV separates native arrays, training float32 preprocessing, and simulated float16.

Selected example NPZ files contain real arrays (load with allow_pickle=False), not PNG-derived values.

![Cutoff retention and magnitude distributions](cutoff_diagnostics.png)

## Training-Space Cutoff Impact

| Split | Type | Pairs | Nonempty | Entirely removed at 0.05 | Nonzero support retained |
|---|---|---:|---:|---:|---:|
| train | overall | 16464 | 9508 | 430 | 33.627% |
| train | consolidation | 2280 | 2280 | 93 | 50.001% |
| train | fluid_overload | 2277 | 2277 | 96 | 47.203% |
| train | none | 6956 | 0 | 0 | NA |
| train | pleural_effusion | 2141 | 2141 | 108 | 37.172% |
| train | pneumothorax | 2810 | 2810 | 133 | 17.979% |
| val | overall | 2108 | 1293 | 56 | 34.088% |
| val | consolidation | 313 | 313 | 5 | 48.058% |
| val | fluid_overload | 309 | 309 | 14 | 49.676% |
| val | none | 815 | 0 | 0 | NA |
| val | pleural_effusion | 294 | 294 | 9 | 38.263% |
| val | pneumothorax | 377 | 377 | 28 | 16.695% |

## Raw GT and Signed-Mask Overlays

### train / pneumothorax / above_005

train_10319_a_1/pair0/variant2

![GT comparison](cases/5f5b553832845ae4.png)

### train / fluid_overload / above_005

train_10158_b_1/pair0/variant0

![GT comparison](cases/68de65eb4600520c.png)

### val / pleural_effusion / above_005

train_10959_a_2/pair0/variant0

![GT comparison](cases/e11ce1e64e2b63fb.png)

### train / none / empty

train_11690_a_2/pair0/variant1

![GT comparison](cases/0a19335c6a4b67ef.png)

### train / fluid_overload / above_005

train_10214_a_1/pair2/variant2

![GT comparison](cases/cd036120fe637a1b.png)

### train / pleural_effusion / above_005

train_11480_a_2/pair1/variant0

![GT comparison](cases/6fb8c6003968d883.png)

### val / fluid_overload / above_005

train_11530_a_2/pair0/variant2

![GT comparison](cases/da64f94be42682e1.png)

### train / pneumothorax / above_005

1.3.6.1.4.1.14519.5.2.1.6279.6001.277662902666135640561346462196/pair2/variant0

![GT comparison](cases/09fa784bc08947ef.png)

### val / none / empty

train_10832_a_2/pair0/variant0

![GT comparison](cases/a7f6c135853ebf65.png)

### train / consolidation / above_005

train_11429_a_2/pair1/variant2

![GT comparison](cases/2d8a52dca33dc3eb.png)

### train / pleural_effusion / above_005

train_11127_a_2/pair2/variant0

![GT comparison](cases/42236ffa96e2ebbe.png)

### val / pneumothorax / above_005

train_11146_a_2/pair2/variant1

![GT comparison](cases/655e61b780d09ea5.png)

### train / consolidation / above_005

train_10722_a_2/pair1/variant0

![GT comparison](cases/cc2ee37a0a8ea59e.png)

### train / consolidation / above_005

train_10438_a_2/pair2/variant0

![GT comparison](cases/4c853dfc29185c88.png)

### train / pleural_effusion / below_005

train_11251_a_2/pair2/variant2

![GT comparison](cases/e187c454dd1ff6be.png)

### val / fluid_overload / below_005

train_10779_a_2/pair0/variant0

![GT comparison](cases/08c64f3448ee8dfd.png)

### val / pleural_effusion / above_005

train_11572_b_2/pair2/variant0

![GT comparison](cases/8e6602c3230a9a72.png)

### val / pneumothorax / above_005

train_11593_a_2/pair2/variant0

![GT comparison](cases/3f0704bfc00bc6bf.png)

### train / fluid_overload / below_005

train_1001_a_1/pair2/variant0

![GT comparison](cases/94344d14336aed98.png)

### val / fluid_overload / above_005

train_10813_a_2/pair0/variant0

![GT comparison](cases/0b2df521c5d8e0b9.png)

### val / fluid_overload / above_005

train_10833_a_2/pair2/variant2

![GT comparison](cases/32e1745576226edb.png)

### val / consolidation / above_005

train_10072_a_2/pair0/variant1

![GT comparison](cases/08f8432b6290e740.png)

### val / consolidation / above_005

train_10014_a_2/pair0/variant2

![GT comparison](cases/9fba925428676db9.png)

### train / consolidation / below_005

train_10464_a_1/pair2/variant2

![GT comparison](cases/b1efb5716d7763c7.png)

### train / pleural_effusion / below_005

train_10627_a_1/pair0/variant2

![GT comparison](cases/aff9a015ce71bbcd.png)

### train / fluid_overload / below_005

train_10186_a_2/pair0/variant0

![GT comparison](cases/c859a4295d6b840c.png)

### train / pneumothorax / below_005

train_10093_a_2/pair0/variant1

![GT comparison](cases/bc7074ded657259d.png)

### train / consolidation / below_005

train_10702_d_1/pair1/variant2

![GT comparison](cases/c2a5c7cfcc9211ce.png)

### val / consolidation / above_005

train_11073_e_1/pair0/variant2

![GT comparison](cases/696d4236f7cd817d.png)

### train / consolidation / below_005

train_10002_a_1/pair1/variant2

![GT comparison](cases/600147343b9b56f4.png)

### val / pleural_effusion / below_005

train_1022_a_3/pair2/variant0

![GT comparison](cases/509a87438eafa43b.png)

### val / pneumothorax / below_005

train_10570_a_2/pair1/variant2

![GT comparison](cases/9583eb0677282679.png)

### train / fluid_overload / above_005

train_11464_a_2/pair1/variant0

![GT comparison](cases/c5a5ede0f7b0b426.png)

### val / consolidation / below_005

volume-51/pair1/variant2

![GT comparison](cases/c30077172c308d3b.png)

### val / fluid_overload / below_005

train_1022_a_5/pair1/variant1

![GT comparison](cases/201c6e642bfeae94.png)

### val / consolidation / below_005

train_10986_a_2/pair2/variant2

![GT comparison](cases/3ab19d90c3539ea4.png)

### val / pneumothorax / below_005

train_1022_a_6/pair0/variant0

![GT comparison](cases/faf9e7e27c1704ca.png)

