# Summary · 3 seed(s): 0, 1, 2

Story-test loss (held-out TinyStories, lower = better) unless the column says steps. Mean ± sample std over seeds; no ± when n = 1. Stage-3 columns use the fresh-pile copy unless they say seen. 'steps to target' = A's after-stage-1 loss + 5%.

| arm | after stage 1 | stage-3 start | final (seen pile) | final (fresh pile) | steps to target | seen − fresh pile at stage-3 start | steps to 3.3 | steps to 3.0 | steps to 2.8 | steps 3.3→2.8 |
|---|---|---|---|---|---|---|---|---|---|---|
| A · stories first | 1.965 ± 0.009 | 5.914 ± 0.187 | 2.364 ± 0.006 | 2.366 ± 0.006 | never | 0.021 ± 0.001 | 25 ± 0 | 46 ± 0 | 69 ± 1 | 44 ± 1 |
| B · code only (no stage 1) | — | 7.090 ± 0.087 | 2.697 ± 0.003 | 2.702 ± 0.006 | never | 0.026 ± 0.001 | 104 ± 2 | 204 ± 1 | 345 ± 5 | 241 ± 3 |
| C · Wikipedia first | 6.813 ± 0.023 | 6.300 ± 0.057 | 2.539 ± 0.002 | 2.542 ± 0.007 | never | 0.023 ± 0.002 | 46 ± 1 | 89 ± 1 | 155 ± 2 | 109 ± 3 |
| S · from scratch (stage 3 only) | — | 10.892 ± 0.027 | 3.153 ± 0.098 | 3.121 ± 0.073 | never | -0.000 ± 0.001 | 402 ± 43 | never | never | never |

## A vs C on 'steps 3.3→2.8' (the clean comparison: same total training, only stage-1 data differs)

- C minus A, per seed: 64, 69, 63 steps
- mean difference 65 steps; seed-to-seed std: A 1.1, C 2.7; difference / larger std = 24.3 (rule of thumb: above ~2 is a finding, below is a hint)

'seen − fresh pile at stage-3 start': loss on the pile A read in stage 1 minus loss on the fresh pile, measured before stage 3. Clearly negative for A (and ~0 for B, C, S) means specific stage-1 text survived stage 2.
