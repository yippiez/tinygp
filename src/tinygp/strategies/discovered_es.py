import numpy as np

from .basic import BasicStrategy
from .basic import BasicStrategyState


class DiscoveredES(BasicStrategy):
    """Learned update-rule evolution strategy.

    Algorithm:
    - Evaluate a population of perturbations around current anchors.
    - Compute summary statistics of fitness and perturbation outcomes.
    - Apply a discovered (hand-designed or learned) update transform to produce the next search step.
    - Generate the next population from updated anchors.

    Intuition:
        Instead of fixed ES formulas, the optimizer follows an update rule tuned to perform well across tasks.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

        adv = float(np.mean(normalized[order[: self.to_k]]) - np.mean(normalized))
        if not self.maximize:
            adv = -adv
        self._learned_mutation = float(np.clip(self._learned_mutation + 0.02 * adv + self._rng.uniform(-0.01, 0.01), 0.01, 1.0))
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
