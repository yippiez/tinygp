import time

import numpy as np
from benchmark.targets import TARGET_REGISTRY

from tinygp.evaluate import eval_uop
from tinygp.evaluate import render_uop
from tinygp.strategies import STRATEGY_REGISTRY


def sanitize(values: np.ndarray) -> np.ndarray:
    """Convert invalid numeric values into bounded finite values."""
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=1e6, posinf=1e6, neginf=-1e6)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error after numeric sanitization."""
    return float(np.mean((sanitize(y_true) - sanitize(y_pred)) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error after numeric sanitization."""
    return float(np.mean(np.abs(sanitize(y_true) - sanitize(y_pred))))


def strategy_kwargs_for(strategy_name: str, simplify_every_n: int = 0) -> dict:
    """Build per-strategy initialization kwargs for benchmarks."""
    kwargs: dict = {}
    if simplify_every_n > 0:
        kwargs["simplify_every_n"] = simplify_every_n

    if strategy_name == "GplearnGP":
        kwargs.update(
            {
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
        )

    return kwargs


def dataset_for_target(target_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate train and test datasets for a target function."""
    assert target_name in TARGET_REGISTRY, f"target must exist in TARGET_REGISTRY: {target_name}"
    target_fn = TARGET_REGISTRY[target_name]
    x_train = np.linspace(-1.0, 1.0, 128)
    x_test = np.linspace(-1.0, 1.0, 256)
    y_train = target_fn(x_train)
    y_test = target_fn(x_test)
    return x_train, y_train, x_test, y_test


def run_strategy(
    strategy_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    generations: int,
    simplify_every_n: int = 0,
):
    """Run one strategy for a fixed number of generations."""
    assert strategy_name in STRATEGY_REGISTRY, f"strategy must exist in STRATEGY_REGISTRY: {strategy_name}"
    strategy_cls = STRATEGY_REGISTRY[strategy_name]

    init_kwargs = {
        "population_size": 256,
        "to_k": 32,
        "mutation_rate": 0.35,
        "crossover_rate": 0.85,
        "max_depth": 5,
        "seed": 0,
        "maximize": True,
    }
    init_kwargs.update(strategy_kwargs_for(strategy_name, simplify_every_n=simplify_every_n))

    strategy = strategy_cls(**init_kwargs)

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
    """Run gplearn symbolic regression for comparison."""
    from gplearn.genetic import SymbolicRegressor

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


def benchmark_strategy(strategy_name: str, target_name: str, generations: int, simplify_every_n: int = 0) -> dict:
    """Benchmark one tinygp strategy on one target."""
    x_train, y_train, x_test, y_test = dataset_for_target(target_name)
    program, expr, train_mse, iter_sec = run_strategy(
        strategy_name,
        x_train,
        y_train,
        generations,
        simplify_every_n=simplify_every_n,
    )
    test_pred = eval_uop(program, x_test)
    return {
        "method": strategy_name,
        "expr": expr,
        "train_mse": train_mse,
        "test_mse": mse(y_test, test_pred),
        "test_mae": mae(y_test, test_pred),
        "iter_sec": iter_sec,
    }


def benchmark_gplearn(target_name: str, generations: int) -> dict:
    """Benchmark gplearn on one target."""
    x_train, y_train, x_test, y_test = dataset_for_target(target_name)
    model, expr, train_mse, iter_sec = run_gplearn(x_train, y_train, generations)
    test_pred = model.predict(x_test.reshape(-1, 1))
    return {
        "method": "gplearn",
        "expr": expr,
        "train_mse": train_mse,
        "test_mse": mse(y_test, test_pred),
        "test_mae": mae(y_test, test_pred),
        "iter_sec": iter_sec,
    }
