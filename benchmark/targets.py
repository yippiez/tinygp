from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class TargetSpec:
    """Describe a benchmark target and its train/test domains."""

    fn: Callable[[np.ndarray], np.ndarray]
    train_min: float = -1.0
    train_max: float = 1.0
    test_min: float = -1.0
    test_max: float = 1.0


def nguyen_1(x: np.ndarray) -> np.ndarray:
    """Return the Nguyen-1 symbolic regression target values."""
    return x**3 + x**2 + x


def nguyen_2(x: np.ndarray) -> np.ndarray:
    """Return the Nguyen-2 symbolic regression target values."""
    return x**4 + x**3 + x**2 + x


def nguyen_3(x: np.ndarray) -> np.ndarray:
    """Return the Nguyen-3 symbolic regression target values."""
    return x**5 + x**4 + x**3 + x**2 + x


def nguyen_4(x: np.ndarray) -> np.ndarray:
    """Return the Nguyen-4 symbolic regression target values."""
    return x**6 + x**5 + x**4 + x**3 + x**2 + x


def nguyen_5(x: np.ndarray) -> np.ndarray:
    """Return the Nguyen-5 symbolic regression target values."""
    return np.sin(x * x) * np.cos(x) - 1.0


def nguyen_6(x: np.ndarray) -> np.ndarray:
    """Return the Nguyen-6 symbolic regression target values."""
    return np.sin(x) + np.sin(x + x * x)


def nguyen_7(x: np.ndarray) -> np.ndarray:
    """Return the Nguyen-7 symbolic regression target values."""
    return np.log(x + 1.0) + np.log(x * x + 1.0)


def nguyen_8(x: np.ndarray) -> np.ndarray:
    """Return the Nguyen-8 symbolic regression target values."""
    return np.sqrt(x)


def koza_1(x: np.ndarray) -> np.ndarray:
    """Return the Koza-1 symbolic regression target values."""
    return x**4 + x**3 + x**2 + x


def koza_2(x: np.ndarray) -> np.ndarray:
    """Return the Koza-2 symbolic regression target values."""
    return x**5 - 2.0 * x**3 + x


def koza_3(x: np.ndarray) -> np.ndarray:
    """Return the Koza-3 symbolic regression target values."""
    return x**6 - 2.0 * x**4 + x**2


def keijzer_1(x: np.ndarray) -> np.ndarray:
    """Return the Keijzer-1 symbolic regression target values."""
    return 0.3 * x * np.sin(2.0 * np.pi * x)


def keijzer_2(x: np.ndarray) -> np.ndarray:
    """Return the Keijzer-2 symbolic regression target values."""
    return 0.3 * x * np.sin(2.0 * np.pi * x)


def keijzer_3(x: np.ndarray) -> np.ndarray:
    """Return the Keijzer-3 symbolic regression target values."""
    return 0.3 * x * np.sin(2.0 * np.pi * x)


TARGET_REGISTRY = {
    "keijzer_1": TargetSpec(keijzer_1, train_min=-1.0, train_max=1.0, test_min=-1.0, test_max=1.0),
    "keijzer_2": TargetSpec(keijzer_2, train_min=-2.0, train_max=2.0, test_min=-2.0, test_max=2.0),
    "keijzer_3": TargetSpec(keijzer_3, train_min=-3.0, train_max=3.0, test_min=-3.0, test_max=3.0),
    "koza_1": TargetSpec(koza_1),
    "koza_2": TargetSpec(koza_2),
    "koza_3": TargetSpec(koza_3),
    "nguyen_1": TargetSpec(nguyen_1),
    "nguyen_2": TargetSpec(nguyen_2),
    "nguyen_3": TargetSpec(nguyen_3),
    "nguyen_4": TargetSpec(nguyen_4),
    "nguyen_5": TargetSpec(nguyen_5),
    "nguyen_6": TargetSpec(nguyen_6),
    "nguyen_7": TargetSpec(nguyen_7, train_min=0.0, train_max=2.0, test_min=0.0, test_max=2.0),
    "nguyen_8": TargetSpec(nguyen_8, train_min=0.0, train_max=4.0, test_min=0.0, test_max=4.0),
}
