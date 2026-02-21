import numpy as np


# NOTE: This NumPy evaluator is faster than Tensor `.numpy()` / `.tolist()`
# materialization in current ad-hoc benchmarks for these GP loops.
def eval_uop(node, x: np.ndarray) -> np.ndarray:
    op_name = node.op.name

    if op_name == "CONST":
        return np.full_like(x, float(node.arg), dtype=np.float64)
    if op_name == "DEFINE_VAR":
        return x
    if op_name == "NEG":
        return -eval_uop(node.src[0], x)

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

    assert False, f"unsupported op in evaluator: {node.op}"
