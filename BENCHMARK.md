# Benchmark

Running all strategies vs gplearn with `generations=5`, `population_size=256`, `to_k=32`, `seed=0`, and train/test grids from `examples/compare_all_vs_gplearn.py` gets the following:

- Iteration count per target: `5`
- Targets per run: `2` (`nguyen_1`, `keijzer_1`)
- Programs evaluated per method per target: `5 * 256 = 1280`
- Programs evaluated per method across both targets: `2560`

| method | nguyen_1 test_mse | keijzer_1 test_mse | avg_test_mse | nguyen_1 iter_sec | keijzer_1 iter_sec |
|---|---:|---:|---:|---:|---:|
| BasicStrategy | 0.018123 | 0.007731 | 0.012927 | 0.01074 | 0.01077 |
| ARS | 0.000391 | 0.027213 | 0.013802 | 0.01422 | 0.00459 |
| ASEBO | 0.002634 | 0.006287 | 0.004461 | 0.02015 | 0.00643 |
| CMA_ES | 0.023940 | 0.008847 | 0.016393 | 0.01670 | 0.04569 |
| CR_FM_NES | 0.040605 | 0.016943 | 0.028774 | 0.00825 | 0.01590 |
| DifferentialEvolution | 0.013862 | 0.017691 | 0.015776 | 0.01969 | 0.02227 |
| DiffusionEvolution | 0.016048 | 0.008952 | 0.012500 | 0.01204 | 0.00990 |
| DiscoveredES | 0.020521 | 0.008062 | 0.014292 | 0.01462 | 0.01977 |
| ESMC | 0.007063 | 0.008730 | 0.007896 | 0.02179 | 0.01560 |
| EvoTF_ES | 0.020521 | 0.008062 | 0.014292 | 0.01476 | 0.01976 |
| GESMR_GA | 0.099786 | 0.027425 | 0.063606 | 0.00754 | 0.01528 |
| GplearnGP | 0.009010 | 0.026926 | 0.017968 | 0.03643 | 0.01919 |
| GradientlessDescent | 0.000391 | 0.027213 | 0.013802 | 0.01457 | 0.00436 |
| GuidedES | 0.013909 | 0.026997 | 0.020453 | 0.01502 | 0.00590 |
| HillClimbing | 0.000391 | 0.027213 | 0.013802 | 0.02031 | 0.00419 |
| LM_MA_ES | 0.004079 | 0.008952 | 0.006515 | 0.01334 | 0.00962 |
| LearnedES | 0.020521 | 0.008062 | 0.014292 | 0.01502 | 0.02515 |
| LearnedGA | 0.061265 | 0.007056 | 0.034160 | 0.01302 | 0.02634 |
| MA_ES | 0.016893 | 0.006255 | 0.011574 | 0.02302 | 0.03230 |
| MR15_GA | 0.033668 | 0.007731 | 0.020699 | 0.00726 | 0.01309 |
| NoiseReuseES | 0.009197 | 0.026764 | 0.017980 | 0.01622 | 0.01257 |
| Open_ES | 0.157365 | 0.027425 | 0.092395 | 0.02193 | 0.03205 |
| PGPE | 0.000391 | 0.027213 | 0.013802 | 0.01362 | 0.01049 |
| PSO | 0.040605 | 0.027256 | 0.033931 | 0.01231 | 0.01371 |
| PersistentES | 0.027077 | 0.026926 | 0.027002 | 0.00780 | 0.00905 |
| RandomSearch | 0.147728 | 0.026043 | 0.086885 | 0.01727 | 0.00968 |
| Rm_ES | 0.016893 | 0.006255 | 0.011574 | 0.01476 | 0.02859 |
| SAMR_GA | 0.058246 | 0.026926 | 0.042586 | 0.00716 | 0.01074 |
| SNES | 0.157315 | 0.008062 | 0.082689 | 0.00860 | 0.02733 |
| SV_CMA_ES | 0.003865 | 0.008016 | 0.005940 | 0.01733 | 0.00866 |
| SV_OpenES | 0.097146 | 0.027165 | 0.062155 | 0.01867 | 0.02402 |
| Sep_CMA_ES | 0.003865 | 0.008952 | 0.006408 | 0.01042 | 0.00943 |
| SimAnneal | 0.002313 | 0.026957 | 0.014635 | 0.01489 | 0.00850 |
| SimpleES | 0.017570 | 0.021849 | 0.019710 | 0.02264 | 0.02401 |
| SimpleGA | 0.057242 | 0.007346 | 0.032294 | 0.00783 | 0.01454 |
| iAMaLGaM_Full | 0.003865 | 0.017097 | 0.010481 | 0.01159 | 0.00982 |
| iAMaLGaM_Univariate | 0.008669 | 0.018755 | 0.013712 | 0.00823 | 0.01947 |
| xNES | 0.040605 | 0.019725 | 0.030165 | 0.01700 | 0.01284 |
| gplearn | 0.003285 | 0.002655 | 0.002970 | 0.09413 | 0.09245 |

## Best strategies vs gplearn (quality + speed)

`avg_iter_sec` is the mean of the two target iteration times. `iter_per_sec = 1 / avg_iter_sec`. `speed_vs_gplearn` uses iteration speed ratio.

| method | avg_test_mse | avg_iter_sec | iter_per_sec | speed_vs_gplearn | total_sec_for_10_iters |
|---|---:|---:|---:|---:|---:|
| gplearn | 0.002970 | 0.09329 | 10.72 | 1.0x | 0.933 |
| ASEBO | 0.004461 | 0.01329 | 75.24 | 7.0x | 0.133 |
| SV_CMA_ES | 0.005940 | 0.01300 | 76.95 | 7.2x | 0.130 |
| Sep_CMA_ES | 0.006408 | 0.00992 | 100.76 | 9.4x | 0.099 |
| LM_MA_ES | 0.006515 | 0.01148 | 87.11 | 8.1x | 0.115 |
| ESMC | 0.007896 | 0.01869 | 53.49 | 5.0x | 0.187 |

## Simplify mode comparison (`simplify_every_n=1`)

This compares each repository strategy in two modes over both targets (`generations=5`):

- Enable periodic simplify by passing `simplify_every_n` when constructing a strategy.
- Example: `BasicStrategy(..., simplify_every_n=1)` simplifies every generation before evaluation.
- Set `simplify_every_n=0` (default) to disable periodic simplify.

- `base_avg_mse`: strategy with no periodic simplify (`simplify_every_n=0`)
- `simplified_avg_mse`: strategy with simplify enabled every generation (`simplify_every_n=1`)
- `delta_mse = simplified_avg_mse - base_avg_mse` (negative is better)
- `simp_vs_base_speed = base_iter_sec / simplified_iter_sec` (less than `1.0` means simplify is slower)

| method | base_avg_mse | simplified_avg_mse | delta_mse | base_iter_sec | simplified_iter_sec | simp_vs_base_speed |
|---|---:|---:|---:|---:|---:|---:|
| ASEBO | 0.004461 | 0.002700 | -0.001760 | 0.01053 | 0.02846 | 0.370 |
| LM_MA_ES | 0.006515 | 0.006015 | -0.000501 | 0.01475 | 0.01991 | 0.741 |
| SV_CMA_ES | 0.005940 | 0.006242 | 0.000302 | 0.00977 | 0.02069 | 0.472 |
| DiffusionEvolution | 0.012500 | 0.006242 | -0.006258 | 0.01087 | 0.02128 | 0.511 |
| NoiseReuseES | 0.017980 | 0.006662 | -0.011318 | 0.01677 | 0.02277 | 0.736 |
| PersistentES | 0.027002 | 0.011349 | -0.015653 | 0.01213 | 0.01613 | 0.752 |
| Sep_CMA_ES | 0.006408 | 0.011776 | 0.005367 | 0.01299 | 0.01944 | 0.668 |
| HillClimbing | 0.013802 | 0.013977 | 0.000175 | 0.01356 | 0.02332 | 0.582 |
| GradientlessDescent | 0.013802 | 0.013977 | 0.000175 | 0.01313 | 0.02353 | 0.558 |
| PGPE | 0.013802 | 0.013977 | 0.000175 | 0.00973 | 0.02722 | 0.357 |
| ARS | 0.013802 | 0.013977 | 0.000175 | 0.00932 | 0.02845 | 0.327 |
| SimAnneal | 0.014635 | 0.014635 | 0.000000 | 0.01558 | 0.02889 | 0.539 |
| iAMaLGaM_Univariate | 0.013712 | 0.015645 | 0.001933 | 0.01383 | 0.01651 | 0.837 |
| ESMC | 0.007896 | 0.019749 | 0.011853 | 0.01574 | 0.02121 | 0.742 |
| Rm_ES | 0.011574 | 0.020115 | 0.008541 | 0.01844 | 0.03765 | 0.490 |
| MA_ES | 0.011574 | 0.020115 | 0.008541 | 0.01961 | 0.03956 | 0.496 |
| LearnedES | 0.014292 | 0.020871 | 0.006579 | 0.02488 | 0.02860 | 0.870 |
| DiscoveredES | 0.014292 | 0.020871 | 0.006579 | 0.01765 | 0.03551 | 0.497 |
| EvoTF_ES | 0.014292 | 0.020871 | 0.006579 | 0.01743 | 0.03596 | 0.485 |
| iAMaLGaM_Full | 0.010481 | 0.020903 | 0.010423 | 0.01084 | 0.01939 | 0.559 |
| DifferentialEvolution | 0.015776 | 0.021477 | 0.005701 | 0.01749 | 0.01975 | 0.885 |
| GplearnGP | 0.017968 | 0.021898 | 0.003930 | 0.02117 | 0.03105 | 0.682 |
| GuidedES | 0.020453 | 0.022034 | 0.001581 | 0.01069 | 0.01530 | 0.699 |
| BasicStrategy | 0.012927 | 0.022575 | 0.009648 | 0.01397 | 0.02059 | 0.679 |
| SimpleES | 0.019710 | 0.026766 | 0.007057 | 0.02281 | 0.02845 | 0.802 |
| GESMR_GA | 0.063606 | 0.027569 | -0.036037 | 0.01164 | 0.02385 | 0.488 |
| PSO | 0.033931 | 0.028353 | -0.005578 | 0.01365 | 0.02485 | 0.549 |
| xNES | 0.030165 | 0.030013 | -0.000152 | 0.01495 | 0.02470 | 0.605 |
| LearnedGA | 0.034160 | 0.031404 | -0.002757 | 0.01927 | 0.03472 | 0.555 |
| CR_FM_NES | 0.028774 | 0.031517 | 0.002743 | 0.01485 | 0.02267 | 0.655 |
| CMA_ES | 0.016393 | 0.032401 | 0.016008 | 0.02700 | 0.05389 | 0.501 |
| SimpleGA | 0.032294 | 0.040444 | 0.008150 | 0.01122 | 0.02293 | 0.490 |
| SV_OpenES | 0.062155 | 0.044095 | -0.018060 | 0.02155 | 0.02941 | 0.733 |
| SNES | 0.082689 | 0.044255 | -0.038434 | 0.01902 | 0.02176 | 0.874 |
| MR15_GA | 0.020699 | 0.062286 | 0.041586 | 0.00985 | 0.01656 | 0.595 |
| RandomSearch | 0.086885 | 0.086885 | 0.000000 | 0.01390 | 0.02508 | 0.554 |
| SAMR_GA | 0.042586 | 0.092145 | 0.049559 | 0.00920 | 0.01949 | 0.472 |
| Open_ES | 0.092395 | 0.092146 | -0.000249 | 0.03130 | 0.02704 | 1.158 |

`gplearn` reference over the same two targets: `avg_test_mse=0.002970`, `avg_iter_sec=0.09358`.
