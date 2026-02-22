import numpy as np

from .basic import BasicStrategy
from .basic import BasicStrategyState


class ASEBO(BasicStrategy):
    """Adaptive ES-Active Subspaces optimizer.

    Algorithm:
    - Rank evaluated candidates and keep top-k elites.
    - Build an active subspace from the strongest elites, using it as a low-dimensional search basis.
    - Seed survivors, then sample parents from that subspace and apply crossover/mutation to generate offspring.
    - Maintain a global best solution over time.

    Intuition:
        ASEBO spends most search effort in directions that recently produced gains, reducing variance in uninformative directions while preserving stochastic exploration.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._subspace = []

    def tell(self, state: BasicStrategyState, fitness: np.ndarray) -> BasicStrategyState:
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

        self._subspace = list(elites[: min(len(elites), 8)])
        next_population = list(elites[: max(1, self.to_k // 2)])
        while len(next_population) < self.population_size:
            source = self._rng.choice(self._subspace)
            child = source
            if self._rng.random() < self.crossover_rate:
                child = self._crossover(source, self._rng.choice(elites))
            if self._rng.random() < self.mutation_rate:
                child = self._mutate(child)
            next_population.append(child)

        return BasicStrategyState(
            generation=state.generation + 1,
            phase="ready",
            population=tuple(next_population),
            best_program=best_program,
            best_fitness=best_fitness,
        )
