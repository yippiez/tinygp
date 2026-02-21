from dataclasses import dataclass
import random

import numpy as np
from tinygrad.tinygrad import UOp, dtypes
from tinygrad.tinygrad.uop import Ops


@dataclass(frozen=True)
class BasicStrategyState:
    generation: int
    phase: str
    population: tuple[UOp, ...]
    best_program: UOp | None
    best_fitness: float | None


class BasicStrategy:
    """Top-k evolutionary strategy using elite selection, crossover, and mutation.

    Each generation keeps the best `to_k` candidates and refills to
    `population_size`. Ranking direction is controlled by `maximize`.
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
    ) -> None:
        if population_size <= 0:
            raise ValueError("population_size must be > 0")
        if to_k <= 0 or to_k > population_size:
            raise ValueError("to_k must be in [1, population_size]")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if const_min > const_max:
            raise ValueError("const_min must be <= const_max")

        self.population_size = population_size
        self.to_k = to_k
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_depth = max_depth
        self.const_min = const_min
        self.const_max = const_max
        self.maximize = maximize
        self._rng = random.Random(seed)
        self._var_x = UOp.variable("x", const_min, const_max, dtype=dtypes.int)

    def ask(self, state: BasicStrategyState | None) -> tuple[list[UOp], BasicStrategyState]:
        """Return the current population to evaluate.

        Args:
            state: Previous strategy state. Pass ``None`` for the first call.

        Returns:
            A tuple containing the population list and the updated state.

        Raises:
            AssertionError: If ``ask`` is called twice in a row without ``tell``.
        """
        if state is None:
            population = tuple(self._random_tree(self.max_depth) for _ in range(self.population_size))
            next_state = BasicStrategyState(
                generation=0,
                phase="asked",
                population=population,
                best_program=None,
                best_fitness=None,
            )
            return list(population), next_state

        assert state.phase == "ready", "ask called twice in a row; call tell after ask"

        next_state = BasicStrategyState(
            generation=state.generation,
            phase="asked",
            population=state.population,
            best_program=state.best_program,
            best_fitness=state.best_fitness,
        )
        return list(state.population), next_state

    def tell(self, state: BasicStrategyState, fitness: np.ndarray) -> BasicStrategyState:
        """Update the strategy using fitness values from the last population.

        Args:
            state: State returned by the preceding ``ask`` call.
            fitness: Fitness array with one score per population member.

        Returns:
            Updated state containing next generation population and best-so-far.

        Raises:
            AssertionError: If call order is invalid or fitness length mismatches population size.
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

        elite_idx = order[: self.to_k]
        elites = [state.population[int(i)] for i in elite_idx]
        next_population = list(elites)

        while len(next_population) < self.population_size:
            if self.to_k > 1 and self._rng.random() < self.crossover_rate:
                parent_a, parent_b = self._rng.sample(elites, 2)
                child = self._crossover(parent_a, parent_b)
            else:
                child = self._rng.choice(elites)

            if self._rng.random() < self.mutation_rate:
                child = self._mutate(child)

            next_population.append(child)

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

        return BasicStrategyState(
            generation=state.generation + 1,
            phase="ready",
            population=tuple(next_population),
            best_program=best_program,
            best_fitness=best_fitness,
        )

    def _random_tree(self, depth: int) -> UOp:
        if depth <= 0 or self._rng.random() < 0.35:
            return self._random_terminal()

        if self._rng.random() < 0.2:
            child = self._random_tree(depth - 1)
            return UOp(Ops.NEG, dtypes.int, (child,))

        lhs = self._random_tree(depth - 1)
        rhs = self._random_tree(depth - 1)
        op = self._rng.choice((Ops.ADD, Ops.SUB, Ops.MUL, Ops.MAX))
        return UOp(op, dtypes.int, (lhs, rhs))

    def _random_terminal(self) -> UOp:
        if self._rng.random() < 0.5:
            return self._var_x
        return UOp.const(dtypes.int, self._rng.randint(self.const_min, self.const_max))

    def _mutate(self, tree: UOp) -> UOp:
        target = self._rng.choice(_collect_nodes(tree))
        replacement_depth = min(self.max_depth, max(1, _tree_depth(target)))
        replacement = self._random_tree(replacement_depth)
        return _replace_subtree(tree, target, replacement)

    def _crossover(self, left: UOp, right: UOp) -> UOp:
        left_target = self._rng.choice(_collect_nodes(left))
        right_source = self._rng.choice(_collect_nodes(right))
        return _replace_subtree(left, left_target, right_source)


def _collect_nodes(root: UOp) -> list[UOp]:
    nodes: list[UOp] = []

    def _walk(node: UOp) -> None:
        nodes.append(node)
        for child in node.src:
            _walk(child)

    _walk(root)
    return nodes


def _tree_depth(root: UOp) -> int:
    if not root.src:
        return 1
    return 1 + max(_tree_depth(child) for child in root.src)


def _replace_subtree(root: UOp, target: UOp, replacement: UOp) -> UOp:
    if root is target:
        return replacement
    if not root.src:
        return root
    new_src = tuple(_replace_subtree(child, target, replacement) for child in root.src)
    if new_src == root.src:
        return root
    return root.replace(src=new_src)
