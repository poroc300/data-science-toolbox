"""Fixtures used in unit tests."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def efsize_arrays():
    """Synthetic arrays for testing effect size calculations."""
    arr1 = np.arange(1, 6)
    arr1_2d = np.column_stack((np.arange(1, 5), np.arange(2, 6)))
    return {
        "arr1_1d": arr1,
        "arr2_1d": arr1 + np.array([1, 2, 2, 1, 4]),
        "arr1_2d": arr1_2d,
        "arr2_2d": arr1_2d + np.repeat([1, 2], repeats=2).reshape(-1, 1),
    }


@pytest.fixture(scope="module")
def ttest_arrays():
    """Synthetic arrays with values normally distributed for t-test functions."""
    rng = np.random.default_rng(seed=42)
    arr1 = rng.normal(loc=10, scale=2, size=50)
    arr2 = rng.normal(loc=12, scale=2, size=50)
    return {"arr1": arr1, "arr2": arr2}
