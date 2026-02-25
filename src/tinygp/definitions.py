from tinygrad.uop import Ops


from dataclasses import dataclass

from tinygrad.uop.ops import UOp


@dataclass(frozen=True)
class StrategyState:
    generation: int
    phase: str
    population: tuple[UOp, ...]
    best_program: UOp | None
    best_fitness: float | None


UNARY_PRIMITIVES: tuple[Ops, ...] = (
    Ops.NEG,
    Ops.SIN,
    Ops.LOG2,
    Ops.EXP2,
    Ops.SQRT,
    Ops.RECIPROCAL,
    Ops.TRUNC,
)

BINARY_PRIMITIVES: tuple[Ops, ...] = (
    Ops.ADD,
    Ops.SUB,
    Ops.MUL,
    Ops.MAX,
    Ops.FDIV,
    Ops.POW,
)
