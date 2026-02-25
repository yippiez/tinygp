import numpy as np
from tinygrad import UOp, dtypes
from tinygrad.uop import Ops

from .basic import BasicStrategy
from .basic import BasicStrategyState
from .basic import _collect_nodes
from .basic import _replace_subtree
from .basic import BINARY_PRIMITIVES
from .basic import UNARY_PRIMITIVES


class GplearnGP(BasicStrategy):
    """gplearn-style GP strategy with tournament and mixed genetic operators.

    This strategy mirrors core gplearn ideas: tournament selection,
    operator-probability dispatch (crossover/subtree/hoist/point/reproduce),
    and light parsimony pressure during parent selection.
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
        tournament_size: int = 20,
        p_subtree_mutation: float | None = None,
        p_hoist_mutation: float | None = None,
        p_point_mutation: float | None = None,
        p_point_replace: float = 0.05,
        parsimony_coefficient: float = 0.001,
        init_method: str = "half_and_half",
    ) -> None:
        super().__init__(
            population_size=population_size,
            to_k=to_k,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            max_depth=max_depth,
            const_min=const_min,
            const_max=const_max,
            seed=seed,
            maximize=maximize,
            simplify_every_n=simplify_every_n,
        )

        assert tournament_size > 0, "tournament_size must be greater than zero"
        assert 0.0 <= p_point_replace <= 1.0, "p_point_replace must be in [0, 1]"
        assert parsimony_coefficient >= 0.0, "parsimony_coefficient must be non-negative"
        assert init_method in {"grow", "full", "half_and_half"}, "init_method must be one of grow/full/half_and_half"

        self.tournament_size = tournament_size
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.init_method = init_method
        self._var_x = UOp.variable("x", float(const_min), float(const_max), dtype=dtypes.float)

        if p_subtree_mutation is None and p_hoist_mutation is None and p_point_mutation is None:
            mutation_budget = min(max(self.mutation_rate, 0.0), max(0.0, 1.0 - self.crossover_rate))
            subtree = 0.5 * mutation_budget
            hoist = 0.25 * mutation_budget
            point = 0.25 * mutation_budget
        else:
            subtree = 0.0 if p_subtree_mutation is None else p_subtree_mutation
            hoist = 0.0 if p_hoist_mutation is None else p_hoist_mutation
            point = 0.0 if p_point_mutation is None else p_point_mutation

        assert subtree >= 0.0, "p_subtree_mutation must be non-negative"
        assert hoist >= 0.0, "p_hoist_mutation must be non-negative"
        assert point >= 0.0, "p_point_mutation must be non-negative"

        total = self.crossover_rate + subtree + hoist + point
        assert total <= 1.0, "crossover and mutation probabilities must sum to <= 1"

        self._method_probs = np.array(
            [self.crossover_rate, self.crossover_rate + subtree, self.crossover_rate + subtree + hoist, total],
            dtype=np.float64,
        )

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
                population.append(self._ramped_program(len(population)))
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
        """Update strategy state from fitness evaluations.

        Args:
            state: State returned by the preceding ``ask`` call.
            fitness: Fitness array with one score per population member.

        Returns:
            Updated state containing next generation population and best-so-far.

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

        lengths = np.array([len(_collect_nodes(program)) for program in state.population], dtype=np.float64)
        if self.maximize:
            penalized = normalized - self.parsimony_coefficient * lengths
            order = np.argsort(normalized)[::-1]
        else:
            penalized = normalized + self.parsimony_coefficient * lengths
            order = np.argsort(normalized)

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
            assert best_program is not None, "best program must be present when best fitness exists"
            assert best_fitness is not None, "best fitness must be present when best program exists"

        next_population = []
        while len(next_population) < self.population_size:
            parent = self._tournament_select(state.population, penalized)
            method = self._rng.random()
            if method < self._method_probs[0]:
                donor = self._tournament_select(state.population, penalized)
                child = self._crossover(parent, donor)
            elif method < self._method_probs[1]:
                child = self._subtree_mutation(parent)
            elif method < self._method_probs[2]:
                child = self._hoist_mutation(parent)
            elif method < self._method_probs[3]:
                child = self._point_mutation(parent)
            else:
                child = parent
            next_population.append(child)

        return BasicStrategyState(
            generation=state.generation + 1,
            phase="ready",
            population=tuple(next_population),
            best_program=best_program,
            best_fitness=best_fitness,
        )

    def _ramped_program(self, index: int) -> UOp:
        min_depth = 2 if self.max_depth >= 2 else 1
        depth = self._rng.randint(min_depth, self.max_depth)
        if self.init_method == "half_and_half":
            method = "full" if index % 2 == 0 else "grow"
        else:
            method = self.init_method
        return self._random_tree_with_method(depth, method)

    def _random_tree_with_method(self, depth: int, method: str) -> UOp:
        if depth <= 1:
            return self._random_terminal()

        if method == "grow" and self._rng.random() < 0.35:
            return self._random_terminal()

        if self._rng.random() < 0.2:
            child = self._random_tree_with_method(depth - 1, method)
            op = self._rng.choice(self._unary_ops)
            return UOp(op, dtypes.float, (child,))

        lhs = self._random_tree_with_method(depth - 1, method)
        rhs = self._random_tree_with_method(depth - 1, method)
        op = self._rng.choice(self._binary_ops)
        return UOp(op, dtypes.float, (lhs, rhs))

    def _random_terminal(self) -> UOp:
        if self._rng.random() < 0.5:
            return self._var_x
        value = self._rng.uniform(float(self.const_min), float(self.const_max))
        return UOp.const(dtypes.float, value)

    def _tournament_select(self, population: tuple[UOp, ...], scores: np.ndarray) -> UOp:
        draw = [self._rng.randrange(len(population)) for _ in range(min(self.tournament_size, len(population)))]
        if self.maximize:
            best_i = max(draw, key=lambda idx: scores[idx])
        else:
            best_i = min(draw, key=lambda idx: scores[idx])
        return population[int(best_i)]

    def _subtree_mutation(self, program: UOp) -> UOp:
        donor = self._ramped_program(self._rng.randrange(self.population_size))
        return self._crossover(program, donor)

    def _hoist_mutation(self, program: UOp) -> UOp:
        nodes = _collect_nodes(program)
        target = self._rng.choice(nodes)
        target_nodes = _collect_nodes(target)
        hoist = self._rng.choice(target_nodes)
        return _replace_subtree(program, target, hoist)

    def _point_mutation(self, program: UOp) -> UOp:
        nodes = _collect_nodes(program)
        targets = [node for node in nodes if self._rng.random() < self.p_point_replace]
        if not targets:
            return program

        child = program
        for target in targets:
            replacement = self._point_replacement(target)
            child = _replace_subtree(child, target, replacement)
        return child

    def _point_replacement(self, node: UOp) -> UOp:
        if node.op in self._unary_ops:
            return UOp(self._rng.choice(self._unary_ops), dtypes.float, node.src)
        if node.op in self._binary_ops:
            return UOp(self._rng.choice(self._binary_ops), dtypes.float, node.src)
        return self._random_terminal()
