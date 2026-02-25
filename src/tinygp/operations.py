from collections.abc import Callable
from typing import TYPE_CHECKING
import random

from tinygrad import UOp, dtypes
from tinygrad.uop import Ops

if TYPE_CHECKING:
    from tinygrad.tensor import Tensor


def uop_primitive_seed_programs(var_x: UOp, unary_ops: tuple[Ops, ...], binary_ops: tuple[Ops, ...]) -> list[UOp]:
    """Generate initial seed programs including variable, constants, and unary/binary operations."""
    programs: list[UOp] = [
        var_x,
        UOp.const(dtypes.float, 1.0),
        UOp.const(dtypes.float, -1.0),
    ]

    for op in unary_ops:
        programs.append(UOp(op, dtypes.float, (var_x,)))

    for op in binary_ops:
        if op is Ops.FDIV:
            rhs = UOp.const(dtypes.float, 0.5)
        elif op is Ops.POW:
            rhs = UOp.const(dtypes.float, 2.0)
        else:
            rhs = UOp.const(dtypes.float, 1.5)
        programs.append(UOp(op, dtypes.float, (var_x, rhs)))

    return programs


def uop_mutate(tree: UOp, rng: random.Random, max_depth: int, random_tree: Callable[[int], UOp]) -> UOp:
    """Mutate a tree by replacing a random node with a new random subtree."""
    target = rng.choice(uop_collect_nodes(tree))
    replacement_depth = min(max_depth, max(1, uop_tree_depth(target)))
    replacement = random_tree(replacement_depth)
    return uop_replace_subtree(tree, target, replacement)


def uop_should_simplify_population(generation: int, simplify_every_n: int) -> bool:
    """Check if the population should be simplified at the current generation."""
    if simplify_every_n <= 0:
        return False
    return generation > 0 and generation % simplify_every_n == 0


def uop_safe_simplify(program: UOp) -> UOp:
    """Simplify a program, returning the original on failure."""
    try:
        return program.simplify()
    except Exception:
        return program


def uop_simplify_population(population: tuple[UOp, ...]) -> tuple[UOp, ...]:
    """Simplify all programs in a population."""
    return tuple(uop_safe_simplify(program) for program in population)


def uop_crossover(left: UOp, right: UOp, rng: random.Random) -> UOp:
    """Perform crossover by replacing a node in left with a node from right."""
    left_target = rng.choice(uop_collect_nodes(left))
    right_source = rng.choice(uop_collect_nodes(right))
    return uop_replace_subtree(left, left_target, right_source)


def uop_collect_nodes(root: UOp) -> list[UOp]:
    """Collect all nodes in a tree, including the root."""
    nodes: list[UOp] = []

    def _walk(node: UOp) -> None:
        nodes.append(node)
        for child in node.src:
            _walk(child)

    _walk(root)
    return nodes


def uop_tree_depth(root: UOp) -> int:
    """Compute the depth of a tree."""
    if not root.src:
        return 1
    return 1 + max(uop_tree_depth(child) for child in root.src)


def uop_replace_subtree(root: UOp, target: UOp, replacement: UOp) -> UOp:
    """Replace a target node with a replacement node in the tree."""
    if root is target:
        return replacement
    if not root.src:
        return root
    new_src = tuple(uop_replace_subtree(child, target, replacement) for child in root.src)
    if new_src == root.src:
        return root
    return root.replace(src=new_src)


def uop_count_nodes(node: UOp) -> int:
    return 1 + sum(uop_count_nodes(child) for child in node.src)


def uop_random_terminal(
    rng: random.Random,
    var_x: UOp,
    const_min: float,
    const_max: float,
) -> UOp:
    """Generate a random terminal node (variable or constant)."""
    if rng.random() < 0.5:
        return var_x
    return UOp.const(dtypes.float, rng.uniform(float(const_min), float(const_max)))


def uop_random_tree(
    depth: int,
    rng: random.Random,
    var_x: UOp,
    unary_ops: tuple[Ops, ...],
    binary_ops: tuple[Ops, ...],
    const_min: float,
    const_max: float,
) -> UOp:
    """Generate a random tree expression with the given maximum depth."""
    if depth <= 0 or rng.random() < 0.35:
        return uop_random_terminal(rng, var_x, const_min, const_max)

    if rng.random() < 0.3:
        child = uop_random_tree(depth - 1, rng, var_x, unary_ops, binary_ops, const_min, const_max)
        op = rng.choice(unary_ops)
        return UOp(op, dtypes.float, (child,))

    lhs = uop_random_tree(depth - 1, rng, var_x, unary_ops, binary_ops, const_min, const_max)
    rhs = uop_random_tree(depth - 1, rng, var_x, unary_ops, binary_ops, const_min, const_max)
    op = rng.choice(binary_ops)
    return UOp(op, dtypes.float, (lhs, rhs))


def uop_tensor_to_tree(tensor: "Tensor", scalar_ranges: dict[str, tuple[float, float]]) -> UOp:
    from tinygrad.tensor import Tensor
    uops_tree, bound_values = tensor.uop.unbind_all()

    for variable in bound_values:
        var_name, var_min, var_max = variable.arg
        assert var_name in scalar_ranges, f"missing scalar range for variable '{var_name}'"
        expected_min, expected_max = scalar_ranges[var_name]
        assert float(var_min) == float(expected_min), (
            f"variable '{var_name}' minimum range mismatch: expected {expected_min}, got {var_min}"
        )
        assert float(var_max) == float(expected_max), (
            f"variable '{var_name}' maximum range mismatch: expected {expected_max}, got {var_max}"
        )

    assert len(bound_values) == len(scalar_ranges), (
        f"range count mismatch: expected {len(scalar_ranges)} variables, got {len(bound_values)}"
    )
    return uops_tree
