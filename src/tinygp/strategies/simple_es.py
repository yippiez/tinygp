import numpy as np

from .basic import BasicStrategy
from .basic import BasicStrategyState


class SimpleES(BasicStrategy):
    """Minimal (mu, lambda)-style evolution strategy.

    Algorithm:
    - Keep top mu elites from the evaluated population.
    - Produce lambda offspring from elites via mutation and optional crossover.
    - Form the next generation from offspring plus optional survivors.
    - Track the global best across generations.

    Intuition:
        SimpleES is a lightweight selection-and-variation loop that emphasizes reliability and low overhead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sigma = 1.0

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

        elite_scores = normalized[order[: self.to_k]]
        threshold = float(np.median(normalized))
        success = float(np.mean(elite_scores >= threshold)) if self.maximize else float(np.mean(elite_scores <= threshold))
        self._sigma = float(np.clip(self._sigma * np.exp(success - 0.2), 0.1, 3.0))
        mutation_rate = float(np.clip(self.mutation_rate * self._sigma, 0.01, 1.0))

        next_population = list(elites[: max(1, self.to_k // 2)])
        while len(next_population) < self.population_size:
            child = self._rng.choice(elites)
            if len(elites) > 1 and self._rng.random() < self.crossover_rate:
                child = self._crossover(child, self._rng.choice(elites))
            if self._rng.random() < mutation_rate:
                child = self._mutate(child)
            next_population.append(child)

        return BasicStrategyState(
            generation=state.generation + 1,
            phase="ready",
            population=tuple(next_population),
            best_program=best_program,
            best_fitness=best_fitness,
        )
