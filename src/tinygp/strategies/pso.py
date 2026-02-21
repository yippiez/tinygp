import numpy as np

from .basic import BasicStrategy
from .basic import BasicStrategyState


class PSO(BasicStrategy):
    """PSO evolutionary strategy for GP populations.

    Uses the shared ask/tell loop and applies algorithm-specific updates
    to build the next generation from evaluated fitness scores.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._personal_best = ()
        self._personal_scores = ()

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

        if len(self._personal_best) != self.population_size:
            self._personal_best = tuple(state.population)
            self._personal_scores = tuple(float(x) for x in normalized.tolist())
        else:
            bests = list(self._personal_best)
            scores_p = list(self._personal_scores)
            for i, score in enumerate(normalized.tolist()):
                better = score > scores_p[i] if self.maximize else score < scores_p[i]
                if better:
                    bests[i] = state.population[i]
                    scores_p[i] = float(score)
            self._personal_best = tuple(bests)
            self._personal_scores = tuple(scores_p)

        global_best = best_program
        next_population = [global_best]
        for i in range(1, self.population_size):
            particle = state.population[i]
            local_best = self._personal_best[i]
            child = particle
            if self._rng.random() < 0.5:
                child = self._crossover(child, local_best)
            if self._rng.random() < 0.5:
                child = self._crossover(child, global_best)
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
