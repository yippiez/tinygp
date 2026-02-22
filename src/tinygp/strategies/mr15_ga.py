import numpy as np

from .basic import BasicStrategy
from .basic import BasicStrategyState


class MR15_GA(BasicStrategy):
    """Mutation-rate-15 style genetic algorithm.

    Algorithm:
    - Keep elites from the current generation.
    - Produce offspring through crossover and mutation under an MR15-inspired mutation schedule.
    - Replace the population with selected survivors and offspring.
    - Preserve global best solution.

    Intuition:
        A fixed, empirically tuned mutation regime offers a simple exploitation/exploration balance without heavy adaptation machinery.
    """

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

        scored = list(zip(state.population, normalized.tolist(), strict=True))
        mutation_rate = 0.15
        next_population = []
        while len(next_population) < self.population_size:
            cand_a = self._rng.sample(scored, min(4, len(scored)))
            cand_b = self._rng.sample(scored, min(4, len(scored)))
            key_fn = (lambda x: x[1]) if self.maximize else (lambda x: -x[1])
            parent_a = max(cand_a, key=key_fn)[0]
            parent_b = max(cand_b, key=key_fn)[0]
            child = self._crossover(parent_a, parent_b) if self._rng.random() < self.crossover_rate else parent_a
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
