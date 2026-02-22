import numpy as np
from tinygp.evaluate import eval_uop
from tinygp.strategies import ASEBO

from tinygrad.tinygrad import UOp
from tinygrad.tinygrad import dtypes
from tinygrad.tinygrad.uop import Ops


_PROTECTED_DIV_EPS = 1e-6
_PROTECTED_LOG_EPS = 1e-6
_EXP2_CLIP = 30.0
_POW_EXP_CLIP = 8.0
_POW_BASE_CLIP = 64.0


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


def render_c_expr(node: UOp) -> str:
    op_name = node.op.name

    if op_name == "CONST":
        value = float(node.arg)
        if not np.isfinite(value):
            if np.isnan(value):
                return "NAN"
            return "INFINITY" if value > 0 else "-INFINITY"
        rounded = round(value)
        if abs(value - rounded) < 1e-9:
            return f"{int(rounded)}.0f"
        return f"{value:.7g}f"

    if op_name == "DEFINE_VAR":
        return str(node.arg[0])

    if op_name == "NEG":
        return f"(-{render_c_expr(node.src[0])})"
    if op_name == "SIN":
        return f"sinf({render_c_expr(node.src[0])})"
    if op_name == "LOG2":
        arg = render_c_expr(node.src[0])
        return f"log2f(fabsf({arg}) + {_PROTECTED_LOG_EPS:.1e}f)"
    if op_name == "EXP2":
        arg = render_c_expr(node.src[0])
        return f"exp2f(fminf(fmaxf({arg}, {-_EXP2_CLIP:.1f}f), {_EXP2_CLIP:.1f}f))"
    if op_name == "SQRT":
        return f"sqrtf(fabsf({render_c_expr(node.src[0])}))"
    if op_name == "RECIPROCAL":
        arg = render_c_expr(node.src[0])
        return f"(fabsf({arg}) > {_PROTECTED_DIV_EPS:.1e}f ? (1.0f / ({arg})) : 0.0f)"
    if op_name == "TRUNC":
        return f"truncf({render_c_expr(node.src[0])})"

    left = render_c_expr(node.src[0])
    right = render_c_expr(node.src[1])

    if op_name == "ADD":
        return f"({left} + {right})"
    if op_name == "SUB":
        return f"({left} - {right})"
    if op_name == "MUL":
        return f"({left} * {right})"
    if op_name == "MAX":
        return f"fmaxf({left}, {right})"
    if op_name == "FDIV":
        return f"(fabsf({right}) > {_PROTECTED_DIV_EPS:.1e}f ? ({left} / {right}) : 1.0f)"
    if op_name == "POW":
        base = f"fminf(fmaxf(fabsf({left}) + {_PROTECTED_LOG_EPS:.1e}f, {_PROTECTED_LOG_EPS:.1e}f), {_POW_BASE_CLIP:.1f}f)"
        exp = f"fminf(fmaxf({right}, {-_POW_EXP_CLIP:.1f}f), {_POW_EXP_CLIP:.1f}f)"
        return f"powf({base}, {exp})"

    assert False, f"unsupported op for C rendering: {node.op}"


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

    reference_c_expr = render_c_expr(reference_kernel)
    optimized_c_expr = render_c_expr(optimized_program)

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


def main() -> None:
    x = np.linspace(-1.0, 1.0, 128)
    reference_kernel = tinygrad_reference_kernel()

    evolve_kernel("tinygrad_poly_kernel", reference_kernel, x)


if __name__ == "__main__":
    main()
