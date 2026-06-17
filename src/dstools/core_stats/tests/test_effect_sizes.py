"""Unit tests for functions in effect_sizes.py."""

import numpy as np
import pytest

from dstools.core_stats import _cohens_d_independent, _cohens_dz_paired, cohens_d


@pytest.fixture(scope="module")
def cohens_d_test_arrays():
    """Fixture to create test arrays for Cohen's d functions."""
    arr1 = np.arange(1, 101)
    arr2 = arr1 + np.random.default_rng(42).normal(0, 0.5, 100)
    arr3 = arr1 * 2
    return arr1, arr2, arr3


def test_cohens_d_independent(cohens_d_test_arrays):
    """Test two-group independent Cohen's d."""
    arr1, arr2, arr3 = cohens_d_test_arrays
    assert _cohens_d_independent(arr1, arr2) < 0.1  # low Cohen's d
    assert abs(_cohens_d_independent(arr1, arr3)) > 0.9  # high Cohen's d


def test_cohens_dz_paired(cohens_d_test_arrays):
    """Test two-group paired Cohen's dz."""
    arr1, arr2, arr3 = cohens_d_test_arrays
    assert _cohens_dz_paired(arr1, arr2) < 0.1  # low Cohen's dz
    assert abs(_cohens_dz_paired(arr1, arr3)) > 0.9  # high Cohen's dz


def test_cohens_dz_invalid_array_sizes():
    """Test Cohen's dz raises error when input arrays have different sizes."""
    with pytest.raises(AssertionError):
        _cohens_dz_paired(np.ones(2), np.ones(1))


@pytest.mark.parametrize(
    "method, cohens_func",
    (("independent", _cohens_d_independent), ("paired", _cohens_dz_paired)),
)
def test_cohens_d(method, cohens_func, cohens_d_test_arrays):
    """Test the public function for Cohen's d."""
    arr1, arr2, _ = cohens_d_test_arrays
    cohens_d_expected = cohens_func(arr1, arr2)
    cohens_d_test = cohens_d(arr1, arr2, method=method)
    assert abs(cohens_d_expected - cohens_d_test) < 0.001
