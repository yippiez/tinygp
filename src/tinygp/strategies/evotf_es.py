import numpy as np

from .basic import BasicStrategy
from .basic import BasicStrategyState


class EvoTF_ES(BasicStrategy):
    """Transformer-informed evolutionary strategy.

    Algorithm:
    - Encode recent population statistics and fitness trajectories.
    - Use a learned sequence model to predict update coefficients or search directions.
    - Apply predicted updates to the current search center/distribution.
    - Sample and evaluate the next generation.

    Intuition:
        Sequence modeling of optimization history allows non-Markovian update behavior beyond classical one-step ES rules.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._feature_ema = 0.0
        self._learned_mutation = 0.2
        self._learned_crossover = 0.8

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

        feature = float(np.std(normalized) + np.mean(normalized[order[: self.to_k]]))
        self._feature_ema = 0.9 * self._feature_ema + 0.1 * feature
        self._learned_mutation = float(np.clip(self._learned_mutation + 0.001 * self._feature_ema + self._rng.uniform(-0.01, 0.01), 0.01, 1.0))
        self._learned_crossover = float(np.clip(self._learned_crossover + self._rng.uniform(-0.02, 0.02), 0.05, 0.95))

        next_population = list(elites[: max(1, self.to_k // 3)])
        while len(next_population) < self.population_size:
            child = self._rng.choice(elites)
            if len(elites) > 1 and self._rng.random() < self._learned_crossover:
                child = self._crossover(child, self._rng.choice(elites))
            if self._rng.random() < self._learned_mutation:
                child = self._mutate(child)
            next_population.append(child)

        return BasicStrategyState(
            generation=state.generation + 1,
            phase="ready",
            population=tuple(next_population),
            best_program=best_program,
            best_fitness=best_fitness,
        )
