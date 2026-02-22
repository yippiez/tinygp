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

Quick snapshot from the current benchmark run (`nguyen_1`, `keijzer_1`):

- `gplearn` reaches the lowest average test MSE (`0.002970`) in this setup.
- `tinygp[ASEBO]` gets close quality (`0.004461`) while running about `7.0x` faster per iteration than `gplearn`.
- `tinygp[SV_CMA_ES]`, `tinygp[Sep_CMA_ES]`, and `tinygp[LM_MA_ES]` also show strong speed/quality balance.

For full benchmark tables, methodology, and simplify-mode comparisons, see `BENCHMARK.md`.

## Allowed GP operations

- `tinygrad` UOps support many operations globally, but this project only evaluates a GP subset.
- `eval_uop` currently supports: `CONST`, `DEFINE_VAR`, `NEG`, `SIN`, `LOG2`, `EXP2`, `SQRT`, `RECIPROCAL`, `TRUNC`, `ADD`, `SUB`, `MUL`, `MAX`, `FDIV`, `POW`.
- First-generation populations now seed at least one instance of each supported primitive op when `population_size` is large enough.
