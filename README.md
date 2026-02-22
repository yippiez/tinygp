# tinygp

A library for genetic programming built on top of tinygrad UOps.

## Basic Usage

```python
import numpy as np
from tinygp.evaluate import eval_uop
from tinygp.strategies import CMA_ES


# Instantiate the search strategy
strategy = CMA_ES(
    population_size=32,
    to_k=8,
    mutation_rate=0.25,
    crossover_rate=0.8,
    max_depth=4,
    seed=0,
    maximize=True,
)

# Target data (example: fit y = x^3 + x^2 + x)
x = np.linspace(-1.0, 1.0, 128)
y = x**3 + x**2 + x

# Initialize state (pass None to ask on first generation)
state = None
num_generations = 100

# Ask-Eval-Tell loop
for i in range(num_generations):
    # Generate a set of candidate programs
    population, state = strategy.ask(state)

    # Evaluate the fitness of the population
    preds = np.stack([eval_uop(program, x) for program in population], axis=0)
    mse = np.mean((preds - y) ** 2, axis=1)
    fitness = -mse  # maximize fitness == minimize MSE

    # Update the evolution strategy
    state = strategy.tell(state, fitness)

# Get best program
assert state is not None, "state must be initialized after at least one generation"
assert state.best_program is not None, "best_program must exist after fitness evaluation"
assert state.best_fitness is not None, "best_fitness must exist after fitness evaluation"
state.best_program, state.best_fitness
```

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

## Implemented Evolution Strategies

| Strategy | Reference | Import | Example |
| --- | --- | --- | --- |
| Simple Evolution Strategy | Rechenberg (1978) | `SimpleES` | `Colab` |
| OpenAI-ES | Salimans et al. (2017) | `Open_ES` | `Colab` |
| CMA-ES | Hansen & Ostermeier (2001) | `CMA_ES` | `Colab` |
| Sep-CMA-ES | Ros & Hansen (2008) | `Sep_CMA_ES` | `Colab` |
| xNES | Wierstra et al. (2014) | `xNES` | `Colab` |
| SNES | Wierstra et al. (2014) | `SNES` | `Colab` |
| MA-ES | Bayer & Sendhoff (2017) | `MA_ES` | `Colab` |
| LM-MA-ES | Loshchilov et al. (2017) | `LM_MA_ES` | `Colab` |
| Rm_ES | Li & Zhang (2017) | `Rm_ES` | `Colab` |
| PGPE | Sehnke et al. (2010) | `PGPE` | `Colab` |
| ARS | Mania et al. (2018) | `ARS` | `Colab` |
| ESMC | Merchant et al. (2021) | `ESMC` | `Colab` |
| Persistent ES | Vicol et al. (2021) | `PersistentES` | `Colab` |
| Noise-Reuse ES | Li et al. (2023) | `NoiseReuseES` | `Colab` |
| CR-FM-NES | Nomura & Ono (2022) | `CR_FM_NES` | `Colab` |
| Guided ES | Maheswaranathan et al. (2018) | `GuidedES` | `Colab` |
| ASEBO | Choromanski et al. (2019) | `ASEBO` | `Colab` |
| Discovered ES | Lange et al. (2023a) | `DiscoveredES` | `Colab` |
| LES | Lange et al. (2023a) | `LearnedES` | `Colab` |
| EvoTF | Lange et al. (2024) | `EvoTF_ES` | `Colab` |
| iAMaLGaM-Full | Bosman et al. (2013) | `iAMaLGaM_Full` | `Colab` |
| iAMaLGaM-Univariate | Bosman et al. (2013) | `iAMaLGaM_Univariate` | `Colab` |
| Gradientless Descent | Golovin et al. (2019) | `GradientlessDescent` | `Colab` |
| Simulated Annealing | Rasdi Rere et al. (2015) | `SimAnneal` | `Colab` |
| Hill Climbing | Rasdi Rere et al. (2015) | `HillClimbing` | `Colab` |
| Random Search | Bergstra & Bengio (2012) | `RandomSearch` | `Colab` |
| SV-CMA-ES | Braun et al. (2024) | `SV_CMA_ES` | `Colab` |
| SV-OpenAI-ES | Liu et al. (2017) | `SV_OpenES` | `Colab` |
| Simple Genetic Algorithm | Such et al. (2017) | `SimpleGA` | `Colab` |
| MR15-GA | Rechenberg (1978) | `MR15_GA` | `Colab` |
| SAMR-GA | Clune et al. (2008) | `SAMR_GA` | `Colab` |
| GESMR-GA | Kumar et al. (2022) | `GESMR_GA` | `Colab` |
| LGA | Lange et al. (2023b) | `LearnedGA` | `Colab` |
| Diffusion Evolution | Zhang et al. (2024) | `DiffusionEvolution` | `Colab` |
| Differential Evolution | Storn & Price (1997) | `DifferentialEvolution` | `Colab` |
| Particle Swarm Optimization | Kennedy & Eberhart (1995) | `PSO` | `Colab` |

## Supported UOPs

tinygp evaluates a focused UOP subset to keep GP search stable, comparable, and efficient.

From tinygrad UOps, tinygp currently supports `ADD`, `CONST`, `DEFINE_VAR`, `EXP2`, `FDIV`, `LOG2`, `MAX`, `MUL`, `NEG`, `POW`, `RECIPROCAL`, `SIN`, `SQRT`, `SUB`, and `TRUNC`; it does not support `AFTER`, `ALLREDUCE`, `AND`, `ASSIGN`, `BARRIER`, `BINARY`, `BIND`, `BITCAST`, `BUFFER`, `BUFFERIZE`, `BUFFER_VIEW`, `CALL`, `CAST`, `CAT`, `CMPEQ`, `CMPLT`, `CMPNE`, `CONTIGUOUS`, `CONTIGUOUS_BACKWARD`, `CONTRACT`, `COPY`, `CUSTOM`, `CUSTOMI`, `DEFINE_LOCAL`, `DEFINE_REG`, `DETACH`, `DEVICE`, `ENCDEC`, `END`, `ENDIF`, `EXPAND`, `FLIP`, `GEP`, `GROUP`, `IDIV`, `IF`, `INDEX`, `INS`, `LINEAR`, `LOAD`, `LUNIQUE`, `MOD`, `MSELECT`, `MSTACK`, `MULACC`, `MULTI`, `NOOP`, `OR`, `PAD`, `PARAM`, `PERMUTE`, `PROGRAM`, `PTRCAT`, `RANGE`, `REDUCE`, `REDUCE_AXIS`, `RESHAPE`, `REWRITE_ERROR`, `SHL`, `SHR`, `SHRINK`, `SINK`, `SOURCE`, `SPECIAL`, `STORE`, `THREEFRY`, `UNIQUE`, `UNROLL`, `VCONST`, `VECTORIZE`, `WHERE`, `WMMA`, and `XOR`.

When `population_size` is large enough, generation 0 is seeded with at least one instance of each supported primitive op.
