# tinygp

A library for genetic programming built on top of tinygrad UOps.

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

## Benchmarks

tinygp includes multiple strategy backends and can be tuned for different quality/speed tradeoffs.

Quick snapshot from the current benchmark run (`14` targets: `nguyen_1..8`, `koza_1..3`, `keijzer_1..3`, `generations=5`), currently compared against `gplearn`:

| Library / Strategy | Avg Test MSE (lower is better) | Speed vs gplearn (higher is better) |
| --- | ---: | ---: |
| `gplearn` | `0.023048` | `1.00x` |
| `tinygp[ASEBO]` | `0.022195` | `5.87x` |
| `tinygp[Sep_CMA_ES]` | `0.026559` | `6.95x` |

The accuracy and speed varias with different configurations. Best way to find optimal strategy is to just test multiple strategies with different parameters for a given problem.

For full benchmark tables, methodology, and simplify-mode comparisons, see `BENCHMARK.md`.

Run all benchmarks with `uv run tinygp-benchmark --strategy all --target all --generations 5`.
Pick a single strategy with `uv run tinygp-benchmark --strategy CMA_ES --target nguyen_1 --generations 5`.

## Examples

- `examples/automatic_kernel_synthesis.py` shows GP-based kernel optimization starting from a tinygrad-defined reference kernel.

## Supported UOPs

tinygp evaluates a focused UOP subset to keep GP search stable, comparable, and efficient.

| UOP Name | Supported |
| --- | --- |
| `ADD` | Yes |
| `AFTER` | No |
| `ALLREDUCE` | No |
| `AND` | No |
| `ASSIGN` | No |
| `BARRIER` | No |
| `BINARY` | No |
| `BIND` | No |
| `BITCAST` | No |
| `BUFFER` | No |
| `BUFFERIZE` | No |
| `BUFFER_VIEW` | No |
| `CALL` | No |
| `CAST` | No |
| `CAT` | No |
| `CMPEQ` | No |
| `CMPLT` | No |
| `CMPNE` | No |
| `CONST` | Yes |
| `CONTIGUOUS` | No |
| `CONTIGUOUS_BACKWARD` | No |
| `CONTRACT` | No |
| `COPY` | No |
| `CUSTOM` | No |
| `CUSTOMI` | No |
| `DEFINE_LOCAL` | No |
| `DEFINE_REG` | No |
| `DEFINE_VAR` | Yes |
| `DETACH` | No |
| `DEVICE` | No |
| `ENCDEC` | No |
| `END` | No |
| `ENDIF` | No |
| `EXP2` | Yes |
| `EXPAND` | No |
| `FDIV` | Yes |
| `FLIP` | No |
| `GEP` | No |
| `GROUP` | No |
| `IDIV` | No |
| `IF` | No |
| `INDEX` | No |
| `INS` | No |
| `LINEAR` | No |
| `LOAD` | No |
| `LOG2` | Yes |
| `LUNIQUE` | No |
| `MAX` | Yes |
| `MOD` | No |
| `MSELECT` | No |
| `MSTACK` | No |
| `MUL` | Yes |
| `MULACC` | No |
| `MULTI` | No |
| `NEG` | Yes |
| `NOOP` | No |
| `OR` | No |
| `PAD` | No |
| `PARAM` | No |
| `PERMUTE` | No |
| `POW` | Yes |
| `PROGRAM` | No |
| `PTRCAT` | No |
| `RANGE` | No |
| `RECIPROCAL` | Yes |
| `REDUCE` | No |
| `REDUCE_AXIS` | No |
| `RESHAPE` | No |
| `REWRITE_ERROR` | No |
| `SHL` | No |
| `SHR` | No |
| `SHRINK` | No |
| `SIN` | Yes |
| `SINK` | No |
| `SOURCE` | No |
| `SPECIAL` | No |
| `SQRT` | Yes |
| `STORE` | No |
| `SUB` | Yes |
| `THREEFRY` | No |
| `TRUNC` | Yes |
| `UNIQUE` | No |
| `UNROLL` | No |
| `VCONST` | No |
| `VECTORIZE` | No |
| `WHERE` | No |
| `WMMA` | No |
| `XOR` | No |

When `population_size` is large enough, generation 0 is seeded with at least one instance of each supported primitive op.
