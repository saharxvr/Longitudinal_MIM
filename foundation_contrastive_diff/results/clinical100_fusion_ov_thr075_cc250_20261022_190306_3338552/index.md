# Clinical100 fusion OV: threshold 0.75, CC >250

Fixed exploratory operating point; not exact Itamar cleanup.
Sign-specific 8-connected components; area scaled from 512x512.
Predictions stay in original image coordinates; annotations are unchanged.
Detection F1 is not pixel Dice. Historical per-pair PAI retains legacy boosts.
F1 with no reference or prediction components is undefined (blank/NA).

| Arm | Mean reader F1 + | Mean reader F1 - | Mean reader F1 all |
|---|---:|---:|---:|
| S6 | 0.4370 | 0.3899 | 0.4173 |
| S9 | 0.4633 | 0.4559 | 0.4601 |
| S12 | 0.4895 | 0.3952 | 0.4474 |
| A | 0.4981 | 0.4420 | 0.4740 |
| R9A | 0.4754 | 0.4076 | 0.4433 |
| B | 0.4962 | 0.4408 | 0.4730 |
| R9B | 0.4354 | 0.4323 | 0.4343 |

- [S6 full OV outputs](arm_S6/protocol.json)
- [S9 full OV outputs](arm_S9/protocol.json)
- [S12 full OV outputs](arm_S12/protocol.json)
- [A full OV outputs](arm_A/protocol.json)
- [R9A full OV outputs](arm_R9A/protocol.json)
- [B full OV outputs](arm_B/protocol.json)
- [R9B full OV outputs](arm_R9B/protocol.json)
