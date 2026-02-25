import numpy as np
import pytest

from tinygp.evaluate import eval_uop
from tinygp.strategies.basic import BasicStrategy


@pytest.fixture
def single_gp_step(target_fn):
    """Fit a single generation using BasicStrategy and return the best program."""
    strategy = BasicStrategy(
        population_size=16,
        to_k=4,
        mutation_rate=0.0,
        crossover_rate=0.0,
        max_depth=2,
        seed=0,
        maximize=True,
    )

    x_train = np.linspace(-3.0, 3.0, 257)
    y_train = target_fn(x_train)

    population, state = strategy.ask(None)
    predictions = np.stack([eval_uop(individual, x_train) for individual in population], axis=0)
    mse = np.mean((predictions - y_train) ** 2, axis=1)
    state = strategy.tell(state, -mse)

    assert state.best_program is not None, "best_program must be tracked after first tell"
    assert state.best_fitness is not None, "best_fitness must be tracked after first tell"
    return state.best_program, state.best_fitness


@pytest.mark.parametrize(
    "name,target_fn",
    [
        ("add", lambda x: x + 1.5),
        ("sub", lambda x: x - 1.5),
        ("mul", lambda x: x * 1.5),
    ],
)
def test_basic_strategy_finds_simple_arithmetic_exactly(name, target_fn, single_gp_step):
    best_program, best_fitness = single_gp_step

    x_test = np.linspace(-5.0, 5.0, 513)
    expected = target_fn(x_test)
    actual = eval_uop(best_program, x_test)

    assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12), f"{name} target must be recovered exactly"
    assert np.isclose(best_fitness, 0.0, atol=1e-12, rtol=0.0), f"{name} should have zero train MSE"
