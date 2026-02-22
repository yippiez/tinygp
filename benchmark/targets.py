import numpy as np


def nguyen_1(x: np.ndarray) -> np.ndarray:
    """Return the Nguyen-1 symbolic regression target values."""
    return x**3 + x**2 + x


def keijzer_1(x: np.ndarray) -> np.ndarray:
    """Return the Keijzer-1 symbolic regression target values."""
    return np.sin(x * x) * np.cos(x) - 1.0


TARGET_REGISTRY = {
    "nguyen_1": nguyen_1,
    "keijzer_1": keijzer_1,
}
