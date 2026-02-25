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

# uv installation 
uv sync
# or install just the package into the active environment
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

| Strategy | Reference | Code |
| --- | --- | --- |
| Simple Evolution Strategy | [Rechenberg (1978)](https://link.springer.com/chapter/10.1007/978-3-642-81283-5_8) | [`SimpleES`](src/tinygp/strategies/simple_es.py) |
| OpenAI-ES | [Salimans et al. (2017)](https://arxiv.org/abs/1703.03864) | [`Open_ES`](src/tinygp/strategies/open_es.py) |
| CMA-ES | [Hansen & Ostermeier (2001)](https://arxiv.org/abs/1604.00772) | [`CMA_ES`](src/tinygp/strategies/cma_es.py) |
| Sep-CMA-ES | [Ros & Hansen (2008)](https://hal.inria.fr/inria-00287367/document) | [`Sep_CMA_ES`](src/tinygp/strategies/sep_cma_es.py) |
| xNES | [Wierstra et al. (2014)](https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf) | [`xNES`](src/tinygp/strategies/xnes.py) |
| SNES | [Wierstra et al. (2014)](https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf) | [`SNES`](src/tinygp/strategies/snes.py) |
| MA-ES | [Bayer & Sendhoff (2017)](https://ieeexplore.ieee.org/document/7875115) | [`MA_ES`](src/tinygp/strategies/ma_es.py) |
| LM-MA-ES | [Loshchilov et al. (2017)](https://arxiv.org/pdf/1705.06693.pdf) | [`LM_MA_ES`](src/tinygp/strategies/lm_ma_es.py) |
| Rm_ES | [Li & Zhang (2017)](https://ieeexplore.ieee.org/document/8080257) | [`Rm_ES`](src/tinygp/strategies/rm_es.py) |
| PGPE | [Sehnke et al. (2010)](https://link.springer.com/chapter/10.1007/978-3-540-87536-9_40) | [`PGPE`](src/tinygp/strategies/pgpe.py) |
| ARS | [Mania et al. (2018)](https://arxiv.org/pdf/1803.07055) | [`ARS`](src/tinygp/strategies/ars.py) |
| ESMC | [Merchant et al. (2021)](https://arxiv.org/abs/2107.09661) | [`ESMC`](src/tinygp/strategies/esmc.py) |
| Persistent ES | [Vicol et al. (2021)](https://arxiv.org/abs/2112.13835) | [`PersistentES`](src/tinygp/strategies/persistent_es.py) |
| Noise-Reuse ES | [Li et al. (2023)](https://arxiv.org/abs/2304.12180) | [`NoiseReuseES`](src/tinygp/strategies/noise_reuse_es.py) |
| CR-FM-NES | [Nomura & Ono (2022)](https://arxiv.org/abs/2201.11422) | [`CR_FM_NES`](src/tinygp/strategies/cr_fm_nes.py) |
| Guided ES | [Maheswaranathan et al. (2018)](https://arxiv.org/abs/1806.10230) | [`GuidedES`](src/tinygp/strategies/guided_es.py) |
| ASEBO | [Choromanski et al. (2019)](https://arxiv.org/abs/1903.04268) | [`ASEBO`](src/tinygp/strategies/asebo.py) |
| Discovered ES | [Lange et al. (2023a)](https://arxiv.org/abs/2211.11260) | [`DiscoveredES`](src/tinygp/strategies/discovered_es.py) |
| LES | [Lange et al. (2023a)](https://arxiv.org/abs/2211.11260) | [`LearnedES`](src/tinygp/strategies/learned_es.py) |
| EvoTF | [Lange et al. (2024)](https://arxiv.org/abs/2403.02985) | [`EvoTF_ES`](src/tinygp/strategies/evotf_es.py) |
| iAMaLGaM-Full | [Bosman et al. (2013)](https://homepages.cwi.nl/~bosman/publications/2013_benchmarkingparameterfree.pdf) | [`iAMaLGaM_Full`](src/tinygp/strategies/iamalgam_full.py) |
| iAMaLGaM-Univariate | [Bosman et al. (2013)](https://homepages.cwi.nl/~bosman/publications/2013_benchmarkingparameterfree.pdf) | [`iAMaLGaM_Univariate`](src/tinygp/strategies/iamalgam_univariate.py) |
| Gradientless Descent | [Golovin et al. (2019)](https://arxiv.org/abs/1911.06317) | [`GradientlessDescent`](src/tinygp/strategies/gradientless_descent.py) |
| Simulated Annealing | [Rasdi Rere et al. (2015)](https://www.sciencedirect.com/science/article/pii/S1877050915035759) | [`SimAnneal`](src/tinygp/strategies/sim_anneal.py) |
| Hill Climbing | [Rasdi Rere et al. (2015)](https://www.sciencedirect.com/science/article/pii/S1877050915035759) | [`HillClimbing`](src/tinygp/strategies/hill_climbing.py) |
| Random Search | [Bergstra & Bengio (2012)](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) | [`RandomSearch`](src/tinygp/strategies/random_search.py) |
| SV-CMA-ES | [Braun et al. (2024)](https://arxiv.org/abs/2410.10390) | [`SV_CMA_ES`](src/tinygp/strategies/sv_cma_es.py) |
| SV-OpenAI-ES | [Liu et al. (2017)](https://arxiv.org/abs/2410.10390) | [`SV_OpenES`](src/tinygp/strategies/sv_openes.py) |
| Simple Genetic Algorithm | [Such et al. (2017)](https://arxiv.org/abs/1712.06567) | [`SimpleGA`](src/tinygp/strategies/simple_ga.py) |
| MR15-GA | [Rechenberg (1978)](https://link.springer.com/chapter/10.1007/978-3-642-81283-5_8) | [`MR15_GA`](src/tinygp/strategies/mr15_ga.py) |
| SAMR-GA | [Clune et al. (2008)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000187) | [`SAMR_GA`](src/tinygp/strategies/samr_ga.py) |
| GESMR-GA | [Kumar et al. (2022)](https://arxiv.org/abs/2204.04817) | [`GESMR_GA`](src/tinygp/strategies/gesmr_ga.py) |
| LGA | [Lange et al. (2023b)](https://arxiv.org/abs/2304.03995) | [`LearnedGA`](src/tinygp/strategies/learned_ga.py) |
| Diffusion Evolution | [Zhang et al. (2024)](https://arxiv.org/pdf/2410.02543) | [`DiffusionEvolution`](src/tinygp/strategies/diffusion_evolution.py) |
| Differential Evolution | [Storn & Price (1997)](https://link.springer.com/article/10.1023/A:1008202821328) | [`DifferentialEvolution`](src/tinygp/strategies/differential_evolution.py) |
| Particle Swarm Optimization | [Kennedy & Eberhart (1995)](https://ieeexplore.ieee.org/document/488968) | [`PSO`](src/tinygp/strategies/pso.py) |

## Supported UOPs

tinygp evaluates a focused UOP subset to keep GP search stable, comparable, and efficient.

From tinygrad UOps, tinygp currently supports `ADD`, `CONST`, `DEFINE_VAR`, `EXP2`, `FDIV`, `LOG2`, `MAX`, `MUL`, `NEG`, `POW`, `RECIPROCAL`, `SIN`, `SQRT`, `SUB`, and `TRUNC`; it does not support `AFTER`, `ALLREDUCE`, `AND`, `ASSIGN`, `BARRIER`, `BINARY`, `BIND`, `BITCAST`, `BUFFER`, `BUFFERIZE`, `BUFFER_VIEW`, `CALL`, `CAST`, `CAT`, `CMPEQ`, `CMPLT`, `CMPNE`, `CONTIGUOUS`, `CONTIGUOUS_BACKWARD`, `CONTRACT`, `COPY`, `CUSTOM`, `CUSTOMI`, `DEFINE_LOCAL`, `DEFINE_REG`, `DETACH`, `DEVICE`, `ENCDEC`, `END`, `ENDIF`, `EXPAND`, `FLIP`, `GEP`, `GROUP`, `IDIV`, `IF`, `INDEX`, `INS`, `LINEAR`, `LOAD`, `LUNIQUE`, `MOD`, `MSELECT`, `MSTACK`, `MULACC`, `MULTI`, `NOOP`, `OR`, `PAD`, `PARAM`, `PERMUTE`, `PROGRAM`, `PTRCAT`, `RANGE`, `REDUCE`, `REDUCE_AXIS`, `RESHAPE`, `REWRITE_ERROR`, `SHL`, `SHR`, `SHRINK`, `SINK`, `SOURCE`, `SPECIAL`, `STORE`, `THREEFRY`, `UNIQUE`, `UNROLL`, `VCONST`, `VECTORIZE`, `WHERE`, `WMMA`, and `XOR`.

When `population_size` is large enough, generation 0 is seeded with at least one instance of each supported primitive op.
