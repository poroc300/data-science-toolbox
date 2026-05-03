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


@pytest.fixture(params=("1d", "2d"), scope="module")
def permutation_test_arrays(request):
    """Fixture that provides both 1D and 2D test arrays for permutation tests."""
    axis = 0
    if request.param == "1d":
        arr1 = np.array([1, 2, 3, 4])
        arr2 = np.array([1, 1, 1, 1])
    else:
        arr1 = np.array([[1, 2], [3, 4]])
        arr2 = np.array([[1, 1], [1, 1]])
    return arr1, arr2, axis
