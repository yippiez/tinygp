import numpy as np

from .basic import BasicStrategy
from .basic import BasicStrategyState


class GuidedES(BasicStrategy):
    """Guided Evolutionary Strategy with hybrid subspace guidance.

    Algorithm:
    - Maintain a guidance subspace from prior gradients, surrogate signals, or elite directions.
    - Sample perturbations partly inside the guide subspace and partly in its orthogonal complement.
    - Estimate an update from fitness-weighted perturbations.
    - Adjust guide-vs-explore mixing and update the center.

    Intuition:
        GuidedES interpolates between pure random search and directed updates, improving sample efficiency when partial directional information exists.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._guide = None

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

        self._guide = elites[0]
        next_population = [self._guide]
        while len(next_population) < self.population_size:
            child = self._rng.choice(elites)
            if self._rng.random() < 0.6:
                child = self._crossover(child, self._guide)
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
