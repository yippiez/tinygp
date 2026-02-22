import numpy as np
from tinygp.evaluate import eval_uop
from tinygp.strategies import ASEBO

from tinygrad.tinygrad import UOp
from tinygrad.tinygrad import dtypes
from tinygrad.tinygrad.uop import Ops


def _b(op: Ops, left: UOp, right: UOp) -> UOp:
    return UOp(op, dtypes.float, (left, right))


def tinygrad_reference_kernel() -> UOp:
    """A deliberately bloated tinygrad kernel equivalent to x^3 + x^2 + x."""
    x = UOp.variable("x", -1.0, 1.0, dtype=dtypes.float)
    one = UOp.const(dtypes.float, 1.0)
    neg_one = UOp.const(dtypes.float, -1.0)
    zero = UOp.const(dtypes.float, 0.0)

    x_plus_one = _b(Ops.ADD, x, one)
    x_minus_one = _b(Ops.ADD, x, neg_one)
    x_sq = _b(Ops.MUL, x, x)

    cubic_arm = _b(Ops.MUL, x, _b(Ops.MUL, x_plus_one, x_minus_one))
    identity_x = _b(Ops.MAX, x, x)
    base = _b(Ops.ADD, cubic_arm, x_sq)
    base = _b(Ops.ADD, base, identity_x)
    base = _b(Ops.ADD, base, _b(Ops.MUL, x, one))

    cancel_1 = _b(Ops.SUB, x, x)
    cancel_2 = _b(Ops.SUB, x_sq, x_sq)
    noisy = _b(Ops.ADD, base, cancel_1)
    noisy = _b(Ops.ADD, noisy, cancel_2)
    return _b(Ops.ADD, noisy, zero)


def count_nodes(node: UOp) -> int:
    return 1 + sum(count_nodes(child) for child in node.src)


def sanitize(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=1e6, posinf=1e6, neginf=-1e6)


def evolve_kernel(kernel_name: str, reference_kernel: UOp, x: np.ndarray, generations: int = 80) -> None:
    y = eval_uop(reference_kernel, x)
    strategy = ASEBO(
        population_size=256,
        to_k=32,
        mutation_rate=0.35,
        crossover_rate=0.85,
        max_depth=6,
        seed=0,
        maximize=True,
        simplify_every_n=4,
    )
    complexity_weight = 1e-4

    state = None
    best_mse = float("inf")

    for gen in range(generations):
        population, state = strategy.ask(state)
        preds = np.stack([eval_uop(individual, x) for individual in population], axis=0)
        mse = np.mean((sanitize(preds) - sanitize(y)) ** 2, axis=1)
        complexity = np.asarray([count_nodes(individual) for individual in population], dtype=np.float64)
        objective = mse + complexity_weight * complexity
        fitness = -objective

        state = strategy.tell(state, fitness)
        current_best = float(np.min(mse))
        best_mse = min(best_mse, current_best)

        if gen % 10 == 0 or gen == generations - 1:
            print(f"{kernel_name} gen={gen:03d} best_mse={best_mse:.8f}")

    assert state is not None, "state must be initialized after at least one generation"
    assert state.best_program is not None, "best_program must exist after fitness evaluation"
    assert state.best_fitness is not None, "best_fitness must exist after fitness evaluation"

    optimized_program = state.best_program.simplify()
    optimized_mse = float(np.mean((sanitize(eval_uop(optimized_program, x)) - sanitize(y)) ** 2))

    reference_c_expr = reference_kernel.render(simplify=False)
    optimized_c_expr = optimized_program.render(simplify=False)

    reference_nodes = count_nodes(reference_kernel)
    optimized_nodes = count_nodes(optimized_program)
    optimized_objective = optimized_mse + complexity_weight * optimized_nodes

    print(f"{kernel_name} strategy=ASEBO")
    print(f"{kernel_name} reference_nodes={reference_nodes}")
    print(f"{kernel_name} optimized_nodes={optimized_nodes}")
    print(f"{kernel_name} node_reduction={reference_nodes - optimized_nodes}")
    print(f"{kernel_name} reference_c_expr={reference_c_expr}")
    print(f"{kernel_name} optimized_c_expr={optimized_c_expr}")
    print(f"{kernel_name} optimized_c_kernel=float {kernel_name}(float x) {{ return {optimized_c_expr}; }}")
    print(f"{kernel_name} optimized_mse={optimized_mse:.8f}")
    print(f"{kernel_name} optimized_objective={optimized_objective:.8f}")
    print(f"{kernel_name} optimized_fitness={state.best_fitness:.8f}")
    print()


x = np.linspace(-1.0, 1.0, 128)
reference_kernel = tinygrad_reference_kernel()
evolve_kernel("tinygrad_poly_kernel", reference_kernel, x)
