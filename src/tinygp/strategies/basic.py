from dataclasses import dataclass
import random

import numpy as np
from tinygrad.tinygrad import UOp, dtypes
from tinygrad.tinygrad.uop import Ops


UNARY_PRIMITIVES: tuple[Ops, ...] = (
    Ops.NEG,
    Ops.SIN,
    Ops.LOG2,
    Ops.EXP2,
    Ops.SQRT,
    Ops.RECIPROCAL,
    Ops.TRUNC,
)

BINARY_PRIMITIVES: tuple[Ops, ...] = (
    Ops.ADD,
    Ops.SUB,
    Ops.MUL,
    Ops.MAX,
    Ops.FDIV,
    Ops.POW,
)


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
            seeds = self._primitive_seed_programs()
            assert self.population_size >= len(seeds), "population_size must be >= primitive seed count for first-generation coverage"
            population = list(seeds)
            while len(population) < self.population_size:
                population.append(self._random_tree(self.max_depth))
            self._rng.shuffle(population)
            next_state = BasicStrategyState(
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
        if self._should_simplify_population(state.generation):
            population = self._simplify_population(state.population)
            if state.best_program is not None:
                best_program = self._safe_simplify(state.best_program)

        next_state = BasicStrategyState(
            generation=state.generation,
            phase="asked",
            population=population,
            best_program=best_program,
            best_fitness=state.best_fitness,
        )
        return list(population), next_state

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

        if self._rng.random() < 0.3:
            child = self._random_tree(depth - 1)
            op = self._rng.choice(self._unary_ops)
            return UOp(op, dtypes.float, (child,))

        lhs = self._random_tree(depth - 1)
        rhs = self._random_tree(depth - 1)
        op = self._rng.choice(self._binary_ops)
        return UOp(op, dtypes.float, (lhs, rhs))

    def _random_terminal(self) -> UOp:
        if self._rng.random() < 0.5:
            return self._var_x
        return UOp.const(dtypes.float, self._rng.uniform(float(self.const_min), float(self.const_max)))

    def _primitive_seed_programs(self) -> list[UOp]:
        programs: list[UOp] = [
            self._var_x,
            UOp.const(dtypes.float, 1.0),
            UOp.const(dtypes.float, -1.0),
        ]

        for op in self._unary_ops:
            programs.append(UOp(op, dtypes.float, (self._var_x,)))

        for op in self._binary_ops:
            if op is Ops.FDIV:
                rhs = UOp.const(dtypes.float, 0.5)
            elif op is Ops.POW:
                rhs = UOp.const(dtypes.float, 2.0)
            else:
                rhs = UOp.const(dtypes.float, 1.5)
            programs.append(UOp(op, dtypes.float, (self._var_x, rhs)))

        return programs

    def _mutate(self, tree: UOp) -> UOp:
        target = self._rng.choice(_collect_nodes(tree))
        replacement_depth = min(self.max_depth, max(1, _tree_depth(target)))
        replacement = self._random_tree(replacement_depth)
        return _replace_subtree(tree, target, replacement)

    def _should_simplify_population(self, generation: int) -> bool:
        if self.simplify_every_n <= 0:
            return False
        return generation > 0 and generation % self.simplify_every_n == 0

    def _safe_simplify(self, program: UOp) -> UOp:
        try:
            return program.simplify()
        except Exception:
            return program

    def _simplify_population(self, population: tuple[UOp, ...]) -> tuple[UOp, ...]:
        return tuple(self._safe_simplify(program) for program in population)

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
