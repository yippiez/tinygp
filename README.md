# tinygp

## Installation

The current recommended way to install tinygp is from source.

### From source

```sh
git clone https://github.com/yippiez/tinygp.git
cd tinygp
# Python3 installation
python3 -m pip install -e .
# For uv
uv pip install -e .
```

### Direct (master)

```sh
# Python3 installation
python3 -m pip install git+https://github.com/yippiez/tinygp.git
# For uv
uv pip install git+https://github.com/yippiez/tinygp.git
```

## Benchmark

tinygp includes multiple strategy backends and can be tuned for different quality/speed tradeoffs versus other GP libraries.

Quick snapshot from the current benchmark run (`14` targets: `nguyen_1..8`, `koza_1..3`, `keijzer_1..3`, `generations=5`):

- `gplearn` remains a strong quality baseline (`avg_test_mse=0.023048`) in this setup.
- `tinygp[ASEBO]` achieves slightly better average quality (`avg_test_mse=0.022195`) while running about `5.87x` faster per iteration than `gplearn`.
- `tinygp[Sep_CMA_ES]` gives strong speed (`6.95x` faster than `gplearn`) with competitive quality (`avg_test_mse=0.026559`).

For full benchmark tables, methodology, and simplify-mode comparisons, see `BENCHMARK.md`.

Run all benchmarks with `uv run tinygp-benchmark --strategy all --target all --generations 5`.
Pick a single strategy with `uv run tinygp-benchmark --strategy CMA_ES --target nguyen_1 --generations 5`.

## Automatic kernel synthesis example

`examples/automatic_kernel_synthesis.py` shows GP-based kernel optimization starting from a tinygrad-defined reference kernel.

- Uses `ASEBO` (one of the stronger benchmarked strategies in this repository).
- Optimizes a deliberately bloated kernel equivalent to `x^3 + x^2 + x`.
- Tracks both fit and compactness via `objective = mse + 1e-4 * node_count`.
- Prints the reference C expression, optimized C expression, and a ready-to-copy C kernel function.

Run it with:

```sh
uv run python examples/automatic_kernel_synthesis.py
```

Typical output includes lines like:

```text
tinygrad_poly_kernel strategy=ASEBO
tinygrad_poly_kernel reference_nodes=35
tinygrad_poly_kernel optimized_nodes=13
tinygrad_poly_kernel node_reduction=22
tinygrad_poly_kernel optimized_mse=0.00000000
tinygrad_poly_kernel optimized_c_kernel=float tinygrad_poly_kernel(float x) { return ...; }
```

## Allowed GP operations

- `tinygrad` UOps support many operations globally, but this project only evaluates a GP subset.
- `eval_uop` currently supports: `CONST`, `DEFINE_VAR`, `NEG`, `SIN`, `LOG2`, `EXP2`, `SQRT`, `RECIPROCAL`, `TRUNC`, `ADD`, `SUB`, `MUL`, `MAX`, `FDIV`, `POW`.
- First-generation populations now seed at least one instance of each supported primitive op when `population_size` is large enough.
