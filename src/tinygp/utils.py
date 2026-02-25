import numpy as np


def sanitize(values: np.ndarray) -> np.ndarray:
    """Convert invalid numeric values into bounded finite values."""
    return np.nan_to_num(np.asarray(values, dtype=np.float64), nan=1e6, posinf=1e6, neginf=-1e6)
