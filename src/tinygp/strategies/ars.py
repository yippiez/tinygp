import random

import numpy as np
from tinygrad import UOp, dtypes

from tinygp.definitions import BINARY_PRIMITIVES, StrategyState, UNARY_PRIMITIVES
from tinygp.operations import (
    uop_crossover,
    uop_mutate,
    uop_primitive_seed_programs,
    uop_random_tree,
    uop_safe_simplify,
    uop_should_simplify_population,
    uop_simplify_population,
)

class ARS:
    """Augmented Random Search.

    Algorithm:
    - Sample symmetric perturbation directions around current anchors.
    - Evaluate positive/negative perturbations and compute directional score differences.
    - Aggregate top directions into an update step with normalized scaling.
    - Move anchors and repeat.

    Intuition:
        ARS estimates a robust descent direction from paired finite differences, often with strong performance-to-complexity tradeoffs.
    """


    def __init__(
        self,
        population_size: int,
        to_k: int,
        *,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.8,
        max_depth: int = 4,
        const_min: int = -8,
        const_max: int = 8,
        seed: int | None = None,
        maximize: bool = True,
        simplify_every_n: int = 0,
        ) -> None:
        assert population_size > 0, "population_size must be > 0"
        assert 0 < to_k <= population_size, "to_k must be in [1, population_size]"
        assert 0.0 <= mutation_rate <= 1.0, "mutation_rate must be in [0, 1]"
        assert 0.0 <= crossover_rate <= 1.0, "crossover_rate must be in [0, 1]"
        assert max_depth >= 1, "max_depth must be >= 1"
        assert const_min <= const_max, "const_min must be <= const_max"
        assert simplify_every_n >= 0, "simplify_every_n must be >= 0"

        self.population_size = population_size
        self.to_k = to_k
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_depth = max_depth
        self.const_min = const_min
        self.const_max = const_max
        self.maximize = maximize
        self.simplify_every_n = simplify_every_n
        self._rng = random.Random(seed)
        self._var_x = UOp.variable("x", float(const_min), float(const_max), dtype=dtypes.float)
        self._unary_ops = UNARY_PRIMITIVES
        self._binary_ops = BINARY_PRIMITIVES


    def ask(self, state: StrategyState | None) -> tuple[list[UOp], StrategyState]:
        """Return the current population to evaluate.

        Args:
            state: Previous strategy state. Pass ``None`` for the first call.

        Returns:
            A tuple containing the population list and the updated state.

        Raises:
            AssertionError: If ``ask`` is called twice in a row without ``tell``.
        """
        if state is None:
            seeds = uop_primitive_seed_programs(self._var_x, self._unary_ops, self._binary_ops)
            assert self.population_size >= len(seeds), "population_size must be >= primitive seed count for first-generation coverage"
            population = list(seeds)
            while len(population) < self.population_size:
                population.append(uop_random_tree(
                    self.max_depth,
                    self._rng,
                    self._var_x,
                    self._unary_ops,
                    self._binary_ops,
                    self.const_min,
                    self.const_max,
                ))
            self._rng.shuffle(population)
            next_state = StrategyState(
                generation=0,
                phase="asked",
                population=tuple(population),
                best_program=None,
                best_fitness=None,
            )
            return population, next_state

        assert state.phase == "ready", "ask called twice in a row; call tell after ask"

        population = state.population
        best_program = state.best_program
        if uop_should_simplify_population(state.generation, self.simplify_every_n):
            population = uop_simplify_population(state.population)
            if state.best_program is not None:
                best_program = uop_safe_simplify(state.best_program)

        next_state = StrategyState(
            generation=state.generation,
            phase="asked",
            population=population,
            best_program=best_program,
            best_fitness=state.best_fitness,
        )
        return list(population), next_state

    def tell(self, state: StrategyState, fitness: np.ndarray) -> StrategyState:
        """Update strategy state from fitness evaluations.

        Args:
            state: State returned by the preceding `ask` call.
            fitness: Fitness array with one score per population member.

        Returns:
            Updated state containing the next generation population.

        Raises:
            AssertionError: If call order is invalid or fitness length mismatches.
        """
        assert state.phase == "asked", "tell called before ask"

        scores = np.asarray(fitness, dtype=float).reshape(-1)
        assert scores.shape[0] == self.population_size, "fitness length must match population_size"

        normalized = np.nan_to_num(
            scores,
            nan=-np.inf if self.maximize else np.inf,
            posinf=np.inf,
            neginf=-np.inf,
        )
        order = np.argsort(normalized)[::-1] if self.maximize else np.argsort(normalized)

        elites = [state.population[int(i)] for i in order[: self.to_k]]
        best_idx = int(order[0])
        current_best_program = state.population[best_idx]
        current_best_fitness = float(normalized[best_idx])

        if state.best_fitness is None:
            best_program = current_best_program
            best_fitness = current_best_fitness
        else:
            improved = current_best_fitness > state.best_fitness if self.maximize else current_best_fitness < state.best_fitness
            best_program = current_best_program if improved else state.best_program
            best_fitness = current_best_fitness if improved else state.best_fitness
            assert best_program is not None, "best program must be present when best_fitness exists"
            assert best_fitness is not None, "best fitness must be present when best_program exists"

        anchor = elites[0]
        next_population = [anchor]
        while len(next_population) < self.population_size:
            pos = uop_mutate(anchor, self._rng, self.max_depth, lambda depth: uop_random_tree(depth, self._rng, self._var_x, self._unary_ops, self._binary_ops, self.const_min, self.const_max))
            neg = uop_mutate(anchor, self._rng, self.max_depth, lambda depth: uop_random_tree(depth, self._rng, self._var_x, self._unary_ops, self._binary_ops, self.const_min, self.const_max))
            next_population.append(pos)
            if len(next_population) < self.population_size:
                next_population.append(neg)

        return StrategyState(
            generation=state.generation + 1,
            phase="ready",
            population=tuple(next_population),
            best_program=best_program,
            best_fitness=best_fitness,
        )
