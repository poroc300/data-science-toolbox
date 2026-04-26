"""Fixtures used in unit tests."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def cohens_d_test_arrays():
    """Fixture to create test arrays for Cohen's d functions."""
    arr1 = np.arange(1, 101)
    arr2 = arr1 + np.random.normal(0, 0.5, 100)
    arr3 = arr1 * 2
    return arr1, arr2, arr3
