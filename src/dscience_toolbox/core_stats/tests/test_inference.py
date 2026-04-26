"""Unit tests for functions in inference.py."""

import numpy as np
import pytest

from dscience_toolbox.core_stats import (
    _cohens_d_independent,
    _cohens_dz_paired,
    cohens_d,
    t_test,
)

# set random seed number
np.random.seed(7)


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


@pytest.mark.parametrize("paired", (True, False))
def test_t_test(paired):
    """Test independent and paired t-test statistics."""
    # create arrays for testing
    arr1 = np.random.normal(loc=10, size=50)
    arr2 = np.random.normal(loc=10, size=50)
    arr3 = np.random.normal(loc=50, size=50)

    # assertions
    _, pval1 = t_test(arr1, arr2, paired=paired)
    _, pval2 = t_test(arr1, arr3, paired=paired)
    assert pval1 > 0.1
    assert pval2 < 0.05
