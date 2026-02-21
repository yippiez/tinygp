import time

import numpy as np
from gplearn.genetic import SymbolicRegressor

from tinygp.benchmark import keijzer_1, nguyen_1
from tinygp.evaluate import eval_uop
from tinygp.evaluate import render_uop
from tinygp.strategies import STRATEGY_REGISTRY


def sanitize(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=1e6, posinf=1e6, neginf=-1e6)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((sanitize(y_true) - sanitize(y_pred)) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(sanitize(y_true) - sanitize(y_pred))))


def run_strategy(strategy_cls, x_train: np.ndarray, y_train: np.ndarray, generations: int, strategy_kwargs: dict | None = None):
    init_kwargs = {
        "population_size": 256,
        "to_k": 32,
        "mutation_rate": 0.35,
        "crossover_rate": 0.85,
        "max_depth": 5,
        "seed": 0,
        "maximize": True,
    }
    if strategy_kwargs is not None:
        init_kwargs.update(strategy_kwargs)

    strategy = strategy_cls(
        **init_kwargs,
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
        render_uop(state.best_program),
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

    strategy_names = ["BasicStrategy"] + sorted(
        key for key in STRATEGY_REGISTRY if key != "BasicStrategy"
    )

    strategy_rows = []
    for strategy_name in strategy_names:
        strategy_cls = STRATEGY_REGISTRY[strategy_name]
        strategy_kwargs: dict | None = None
        if strategy_name == "GplearnGP":
            strategy_kwargs = {
                "const_min": -2,
                "const_max": 2,
                "max_depth": 6,
                "tournament_size": 20,
                "p_subtree_mutation": 0.05,
                "p_hoist_mutation": 0.05,
                "p_point_mutation": 0.05,
                "p_point_replace": 0.1,
                "parsimony_coefficient": 0.0,
            }
        program, expr, train_mse, iter_sec = run_strategy(
            strategy_cls,
            x_train,
            y_train,
            generations,
            strategy_kwargs=strategy_kwargs,
        )
        test_pred = eval_uop(program, x_test)
        strategy_rows.append(
            (
                strategy_name,
                expr,
                train_mse,
                mse(y_test, test_pred),
                mae(y_test, test_pred),
                iter_sec,
            )
        )

    gpl_model, gpl_expr, gpl_train_mse, gpl_iter_sec = run_gplearn(x_train, y_train, generations)
    gpl_test_pred = gpl_model.predict(x_test.reshape(-1, 1))

    gpl_test_mse = mse(y_test, gpl_test_pred)
    gpl_test_mae = mae(y_test, gpl_test_pred)

    print(f"benchmark: {name}")
    print("method                 train_mse    test_mse     test_mae     iter_sec")
    for row in strategy_rows:
        strategy_name, _, train_mse, test_mse, test_mae, iter_sec = row
        print(f"{strategy_name:20s} {train_mse:10.6f}  {test_mse:10.6f}  {test_mae:10.6f}  {iter_sec:9.5f}")
    print(
        f"{'gplearn':20s} {gpl_train_mse:10.6f}  {gpl_test_mse:10.6f}  {gpl_test_mae:10.6f}  {gpl_iter_sec:9.5f}"
    )
    for row in strategy_rows:
        strategy_name, expr, *_ = row
        print(f"{strategy_name} expr: {expr}")
    print(f"gplearn expr: {gpl_expr}")
    print()


def main() -> None:
    compare_on_target("nguyen_1", nguyen_1)
    compare_on_target("keijzer_1", keijzer_1)


if __name__ == "__main__":
    main()
