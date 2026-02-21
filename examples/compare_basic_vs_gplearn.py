import time

import numpy as np
from gplearn.genetic import SymbolicRegressor

from tinygp.benchmark import keijzer_1, nguyen_1
from tinygp.evaluate import eval_uop
from tinygp.strategies import BasicStrategy


def sanitize(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=1e6, posinf=1e6, neginf=-1e6)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((sanitize(y_true) - sanitize(y_pred)) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(sanitize(y_true) - sanitize(y_pred))))


def run_basic(x_train: np.ndarray, y_train: np.ndarray, generations: int):
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
    iteration_times: list[float] = []

    for _ in range(generations):
        start = time.perf_counter()
        population, state = strategy.ask(state)
        preds = np.stack([eval_uop(individual, x_train) for individual in population], axis=0)
        fitness = -np.mean((sanitize(preds) - sanitize(y_train)) ** 2, axis=1)
        state = strategy.tell(state, fitness)
        iteration_times.append(time.perf_counter() - start)

    assert state is not None, "state must exist after running generations"
    assert state.best_program is not None, "best program must be tracked in state"
    assert state.best_fitness is not None, "best fitness must be tracked in state"

    return (
        state.best_program,
        state.best_program.simplify().render(),
        float(-state.best_fitness),
        float(np.mean(iteration_times)),
    )


def run_gplearn(x_train: np.ndarray, y_train: np.ndarray, generations: int):
    model = SymbolicRegressor(
        population_size=256,
        generations=generations,
        tournament_size=20,
        stopping_criteria=0.0,
        p_crossover=0.85,
        p_subtree_mutation=0.05,
        p_hoist_mutation=0.05,
        p_point_mutation=0.05,
        max_samples=1.0,
        function_set=("add", "sub", "mul", "div", "sin", "cos", "max"),
        parsimony_coefficient=0.001,
        random_state=0,
        metric="mse",
    )

    start = time.perf_counter()
    model.fit(x_train.reshape(-1, 1), y_train)
    total_time = time.perf_counter() - start
    train_pred = model.predict(x_train.reshape(-1, 1))

    return model, str(model._program), mse(y_train, train_pred), total_time / generations


def compare_on_target(name: str, target_fn, generations: int = 40) -> None:
    x_train = np.linspace(-1.0, 1.0, 128)
    x_test = np.linspace(-1.0, 1.0, 256)
    y_train = target_fn(x_train)
    y_test = target_fn(x_test)

    basic_program, basic_expr, basic_train_mse, basic_iter_sec = run_basic(x_train, y_train, generations)
    gpl_model, gpl_expr, gpl_train_mse, gpl_iter_sec = run_gplearn(x_train, y_train, generations)

    basic_test_pred = eval_uop(basic_program, x_test)
    gpl_test_pred = gpl_model.predict(x_test.reshape(-1, 1))

    basic_test_mse = mse(y_test, basic_test_pred)
    gpl_test_mse = mse(y_test, gpl_test_pred)
    basic_test_mae = mae(y_test, basic_test_pred)
    gpl_test_mae = mae(y_test, gpl_test_pred)

    print(f"benchmark: {name}")
    print("method     train_mse    test_mse     test_mae     iter_sec")
    print(
        f"basic      {basic_train_mse:10.6f}  {basic_test_mse:10.6f}  {basic_test_mae:10.6f}  {basic_iter_sec:9.5f}"
    )
    print(
        f"gplearn    {gpl_train_mse:10.6f}  {gpl_test_mse:10.6f}  {gpl_test_mae:10.6f}  {gpl_iter_sec:9.5f}"
    )
    print(f"basic expr: {basic_expr}")
    print(f"gplearn expr: {gpl_expr}")
    print()


def main() -> None:
    compare_on_target("nguyen_1", nguyen_1)
    compare_on_target("keijzer_1", keijzer_1)


if __name__ == "__main__":
    main()
