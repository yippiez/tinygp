import numpy as np

from tinygp.benchmark import keijzer_1, nguyen_1
from tinygp.evaluate import eval_uop
from tinygp.strategies import BasicStrategy


def evolve_target(name: str, target_fn, x: np.ndarray, generations: int = 60) -> None:
    y = target_fn(x)
    strategy = BasicStrategy(
        population_size=256,
        to_k=32,
        mutation_rate=0.35,
        crossover_rate=0.85,
        max_depth=5,
        seed=0,
        maximize=True,
    )

    state = None
    best_mse = float("inf")

    for gen in range(generations):
        population, state = strategy.ask(state)
        preds = np.stack([eval_uop(individual, x) for individual in population], axis=0)
        mse = np.mean((preds - y) ** 2, axis=1)
        fitness = -mse

        state = strategy.tell(state, fitness)
        current_best = float(np.min(mse))
        best_mse = min(best_mse, current_best)

        if gen % 10 == 0 or gen == generations - 1:
            print(f"{name} gen={gen:03d} best_mse={best_mse:.6f}")

    assert state is not None, "state must be initialized after at least one generation"
    assert state.best_program is not None, "best_program must exist after fitness evaluation"
    assert state.best_fitness is not None, "best_fitness must exist after fitness evaluation"
    best_expr = state.best_program.simplify().render()
    print(f"{name} best_fitness={state.best_fitness:.6f}")
    print(f"{name} best_c_expr={best_expr}")
    print()


def main() -> None:
    x_nguyen = np.linspace(-1.0, 1.0, 128)
    x_keijzer = np.linspace(-1.0, 1.0, 128)

    evolve_target("nguyen_1", nguyen_1, x_nguyen)
    evolve_target("keijzer_1", keijzer_1, x_keijzer)


if __name__ == "__main__":
    main()
