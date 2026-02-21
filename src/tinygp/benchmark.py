import numpy as np


def nguyen_1(x: np.ndarray) -> np.ndarray:
    """Cubic polynomial target used as an easy symbolic-regression landscape.

    This function defines a smooth single-basin style objective where GP can
    recover a compact algebraic expression.
    """
    return x**3 + x**2 + x


def keijzer_1(x: np.ndarray) -> np.ndarray:
    """Oscillatory trigonometric target for symbolic-regression search.

    This function creates a more rugged fitness landscape than a plain
    polynomial because of periodic nonlinear interactions.
    """
    return np.sin(x * x) * np.cos(x) - 1.0
