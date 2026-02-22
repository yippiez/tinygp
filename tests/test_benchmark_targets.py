import numpy as np

from benchmark.common import dataset_for_target
from benchmark.targets import TARGET_REGISTRY


def test_target_registry_has_expanded_suite() -> None:
    expected_targets = {
        "keijzer_1",
        "keijzer_2",
        "keijzer_3",
        "koza_1",
        "koza_2",
        "koza_3",
        "nguyen_1",
        "nguyen_2",
        "nguyen_3",
        "nguyen_4",
        "nguyen_5",
        "nguyen_6",
        "nguyen_7",
        "nguyen_8",
    }
    assert expected_targets.issubset(TARGET_REGISTRY), "expanded target suite must be present in TARGET_REGISTRY"


def test_dataset_generation_produces_finite_values_for_all_targets() -> None:
    for target_name in sorted(TARGET_REGISTRY):
        x_train, y_train, x_test, y_test = dataset_for_target(target_name)
        assert x_train.shape == (128,), f"{target_name} train inputs must have shape (128,)"
        assert y_train.shape == (128,), f"{target_name} train targets must have shape (128,)"
        assert x_test.shape == (256,), f"{target_name} test inputs must have shape (256,)"
        assert y_test.shape == (256,), f"{target_name} test targets must have shape (256,)"
        assert np.all(np.isfinite(y_train)), f"{target_name} train targets must be finite"
        assert np.all(np.isfinite(y_test)), f"{target_name} test targets must be finite"
