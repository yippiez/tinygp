import numpy as np


# NOTE: This NumPy evaluator is faster than Tensor `.numpy()` / `.tolist()`
# materialization in current ad-hoc benchmarks for these GP loops.
_PROTECTED_DIV_EPS = 1e-6
_PROTECTED_LOG_EPS = 1e-6
_EXP2_CLIP = 30.0
_POW_EXP_CLIP = 8.0
_POW_BASE_CLIP = 64.0


def eval_uop(node, x: np.ndarray) -> np.ndarray:
    op_name = node.op.name

    if op_name == "CONST":
        return np.full_like(x, float(node.arg), dtype=np.float64)
    if op_name == "DEFINE_VAR":
        return x
    if op_name == "NEG":
        return -eval_uop(node.src[0], x)
    if op_name == "SIN":
        return np.sin(eval_uop(node.src[0], x))
    if op_name == "LOG2":
        return np.log2(np.abs(eval_uop(node.src[0], x)) + _PROTECTED_LOG_EPS)
    if op_name == "EXP2":
        return np.exp2(np.clip(eval_uop(node.src[0], x), -_EXP2_CLIP, _EXP2_CLIP))
    if op_name == "SQRT":
        return np.sqrt(np.abs(eval_uop(node.src[0], x)))
    if op_name == "RECIPROCAL":
        value = eval_uop(node.src[0], x)
        mask = np.abs(value) > _PROTECTED_DIV_EPS
        out = np.zeros_like(value, dtype=np.float64)
        np.divide(1.0, value, out=out, where=mask)
        return out
    if op_name == "TRUNC":
        return np.trunc(eval_uop(node.src[0], x))

    left = eval_uop(node.src[0], x)
    right = eval_uop(node.src[1], x)

    if op_name == "ADD":
        return left + right
    if op_name == "SUB":
        return left - right
    if op_name == "MUL":
        return left * right
    if op_name == "MAX":
        return np.maximum(left, right)
    if op_name == "FDIV":
        mask = np.abs(right) > _PROTECTED_DIV_EPS
        out = np.ones_like(left, dtype=np.float64)
        np.divide(left, right, out=out, where=mask)
        return out
    if op_name == "POW":
        base = np.clip(np.abs(left) + _PROTECTED_LOG_EPS, _PROTECTED_LOG_EPS, _POW_BASE_CLIP)
        exp = np.clip(right, -_POW_EXP_CLIP, _POW_EXP_CLIP)
        with np.errstate(over="ignore", invalid="ignore"):
            return np.power(base, exp)

    assert False, f"unsupported op in evaluator: {node.op}"


def render_uop(node) -> str:
    op_name = node.op.name

    if op_name == "CONST":
        value = float(node.arg)
        if not np.isfinite(value):
            if np.isnan(value):
                return "nan"
            return "inf" if value > 0 else "-inf"
        rounded = round(value)
        if abs(value - rounded) < 1e-9:
            return str(int(rounded))
        return f"{value:.6g}"

    if op_name == "DEFINE_VAR":
        return str(node.arg[0])

    if op_name == "NEG":
        return f"(-{render_uop(node.src[0])})"
    if op_name == "SIN":
        return f"sin({render_uop(node.src[0])})"
    if op_name == "LOG2":
        return f"log2abs({render_uop(node.src[0])})"
    if op_name == "EXP2":
        return f"exp2({render_uop(node.src[0])})"
    if op_name == "SQRT":
        return f"sqrtabs({render_uop(node.src[0])})"
    if op_name == "RECIPROCAL":
        return f"recip({render_uop(node.src[0])})"
    if op_name == "TRUNC":
        return f"trunc({render_uop(node.src[0])})"

    left = render_uop(node.src[0])
    right = render_uop(node.src[1])

    if op_name == "ADD":
        return f"({left} + {right})"
    if op_name == "SUB":
        return f"({left} - {right})"
    if op_name == "MUL":
        return f"({left} * {right})"
    if op_name == "MAX":
        return f"max({left}, {right})"
    if op_name == "FDIV":
        return f"pdiv({left}, {right})"
    if op_name == "POW":
        return f"powabs({left}, {right})"

    return str(node)
