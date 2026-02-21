import numpy as np


def keijzer_1(x: np.ndarray) -> np.ndarray:
    return np.sin(x * x) * np.cos(x) - 1.0
