import numpy as np

from .basic import BasicStrategy
from .basic import BasicStrategyState


class ESMC(BasicStrategy):
    """ESMC evolutionary strategy for GP populations.

    Uses the shared ask/tell loop and applies algorithm-specific updates
    to build the next generation from evaluated fitness scores.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._archive = []

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

        rank_weights = np.linspace(1.0, 0.1, self.to_k)
        parent_count = max(self.to_k, 4)
        parent_idxs = self._rng.choices(range(self.to_k), weights=rank_weights.tolist(), k=parent_count)
        parents = [elites[i] for i in parent_idxs]
        parents.extend(self._archive[-4:])

        next_population = list(elites[: max(1, self.to_k // 3)])
        while len(next_population) < self.population_size:
            child = self._rng.choice(parents)
            if len(parents) > 1 and self._rng.random() < self.crossover_rate:
                child = self._crossover(child, self._rng.choice(parents))
            if self._rng.random() < self.mutation_rate:
                child = self._mutate(child)
            next_population.append(child)
        self._archive.extend(elites[:2])
        self._archive = self._archive[-16:]

        return BasicStrategyState(
            generation=state.generation + 1,
            phase="ready",
            population=tuple(next_population),
            best_program=best_program,
            best_fitness=best_fitness,
        )
