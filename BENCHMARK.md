# Benchmark

This benchmark snapshot expands the suite from 2 targets to 14 targets and runs all registered tinygp strategies plus `gplearn`.

- Command: `uv run tinygp-benchmark --strategy all --target all --generations 5`
- Generations per run: `5`
- Population size: `256`
- Selection size (`to_k`): `32`
- Seed: `0`
- Train/test points per target: `128 / 256`
- Targets per run: `14`
- Methods per run: `39` (`38` tinygp + `gplearn`)
- Programs evaluated per method per target: `5 * 256 = 1280`
- Programs evaluated per method across all targets: `17920`

## Target set

- `keijzer_1` (`x in [-1, 1]`): `0.3 * x * sin(2*pi*x)`
- `keijzer_2` (`x in [-2, 2]`): `0.3 * x * sin(2*pi*x)`
- `keijzer_3` (`x in [-3, 3]`): `0.3 * x * sin(2*pi*x)`
- `koza_1`: `x^4 + x^3 + x^2 + x`
- `koza_2`: `x^5 - 2*x^3 + x`
- `koza_3`: `x^6 - 2*x^4 + x^2`
- `nguyen_1`: `x^3 + x^2 + x`
- `nguyen_2`: `x^4 + x^3 + x^2 + x`
- `nguyen_3`: `x^5 + x^4 + x^3 + x^2 + x`
- `nguyen_4`: `x^6 + x^5 + x^4 + x^3 + x^2 + x`
- `nguyen_5`: `sin(x^2) * cos(x) - 1`
- `nguyen_6`: `sin(x) + sin(x + x^2)`
- `nguyen_7` (`x in [0, 2]`): `log(x + 1) + log(x^2 + 1)`
- `nguyen_8` (`x in [0, 4]`): `sqrt(x)`

## Best average quality (all 14 targets)

`avg_test_mse` and `avg_iter_sec` are averaged over all 14 targets.

| method | avg_test_mse | avg_iter_sec | speed_vs_gplearn |
|---|---:|---:|---:|
| GradientlessDescent | 0.020428 | 0.01567 | 6.58x |
| HillClimbing | 0.020428 | 0.01893 | 5.45x |
| PGPE | 0.020428 | 0.01528 | 6.75x |
| ARS | 0.021034 | 0.01691 | 6.10x |
| ASEBO | 0.022195 | 0.01756 | 5.87x |
| gplearn | 0.023048 | 0.10315 | 1.00x |
| GplearnGP | 0.024063 | 0.02166 | 4.76x |
| ESMC | 0.024948 | 0.01816 | 5.68x |
| MA_ES | 0.025651 | 0.02664 | 3.87x |
| Rm_ES | 0.025651 | 0.02896 | 3.56x |
| Sep_CMA_ES | 0.026559 | 0.01484 | 6.95x |
| BasicStrategy | 0.027534 | 0.01574 | 6.55x |

## Selected per-target quality

`test_mse` for a representative set of methods over every target:

| target | gplearn | ASEBO | Sep_CMA_ES | SV_CMA_ES | LM_MA_ES | BasicStrategy |
|---|---:|---:|---:|---:|---:|---:|
| keijzer_1 | 0.012300 | 0.009551 | 0.009800 | 0.009797 | 0.009641 | 0.009413 |
| keijzer_2 | 0.057126 | 0.047821 | 0.052092 | 0.049997 | 0.049309 | 0.046203 |
| keijzer_3 | 0.131769 | 0.090873 | 0.101615 | 0.120686 | 0.123103 | 0.122585 |
| koza_1 | 0.001882 | 0.013918 | 0.031325 | 0.017541 | 0.022014 | 0.011128 |
| koza_2 | 0.013945 | 0.018023 | 0.019750 | 0.020847 | 0.019262 | 0.020184 |
| koza_3 | 0.004288 | 0.001739 | 0.002732 | 0.002200 | 0.002567 | 0.002281 |
| nguyen_1 | 0.003285 | 0.002634 | 0.003865 | 0.003865 | 0.004079 | 0.018123 |
| nguyen_2 | 0.001882 | 0.013918 | 0.031325 | 0.017541 | 0.022014 | 0.011128 |
| nguyen_3 | 0.010013 | 0.000301 | 0.030796 | 0.062707 | 0.051790 | 0.047989 |
| nguyen_4 | 0.045927 | 0.010715 | 0.021711 | 0.088140 | 0.027087 | 0.019359 |
| nguyen_5 | 0.002655 | 0.006287 | 0.008952 | 0.008016 | 0.008952 | 0.007731 |
| nguyen_6 | 0.020407 | 0.093597 | 0.036235 | 0.041664 | 0.096405 | 0.047727 |
| nguyen_7 | 0.002786 | 0.001357 | 0.021630 | 0.021630 | 0.021630 | 0.021630 |
| nguyen_8 | 0.014403 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Notes

- `MA_ES` and `Rm_ES` still emit non-fatal runtime warnings from sigma updates (`overflow encountered in exp`) during some runs.
- `koza_1` and `nguyen_2` intentionally use the same polynomial form, so those rows are expected to match closely.
