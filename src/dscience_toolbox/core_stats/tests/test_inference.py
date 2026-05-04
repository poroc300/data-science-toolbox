"""Unit tests for functions in inference.py."""

from contextlib import suppress

import numpy as np
import pytest

from dscience_toolbox.core_stats import (
    _cohens_d_independent,
    _cohens_dz_paired,
    _permutation_stat_independent,
    _permutation_stat_paired,
    check_normality,
    cohens_d,
    permutation_test_means,
    rank_based_tests,
    t_test,
)

# set random seed number
np.random.seed(7)


def test_check_normality_normal_dist():
    """Test when input array exhibits normal distribution."""
    # create test array and get result
    arr = np.random.normal(50, 5, size=100)
    test = check_normality(arr)

    # check p-value is > 0.1 for normality tests
    for test_key, values in test.items():
        with suppress(IndexError):  # will skip kurtosis and skewness
            assert test[test_key]["pval"] > 0.1

    # skewness and kurtosis must be close to zero
    assert abs(test["skewness"]) < 0.1
    assert abs(test["kurtosis"]) < 0.1


def test_check_normality_exponential_dist():
    """Test when input array exhibits a non-normal distribution."""
    # create test array and get result
    arr = np.random.exponential(50, size=100)
    test = check_normality(arr)

    # check p-value is < 0.05 for normality tests
    for test_key, values in test.items():
        with suppress(IndexError):  # will skip kurtosis and skewness
            assert test[test_key]["pval"] < 0.05

    # skewness and kurtosis must be higher
    assert abs(test["skewness"]) > 0.1
    assert abs(test["kurtosis"]) > 0.1


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


@pytest.mark.parametrize("paired", (True, False))
def test_rank_based_tests(paired):
    """Test independent and paired rank-based statistics."""
    # create arrays for testing
    arr1 = np.random.randint(1, 3, size=50)
    arr2 = arr1 + np.tile([-1, 1], 25)
    arr3 = np.arange(1, 51)

    # assertions
    _, pval1 = rank_based_tests(arr1, arr2, paired=paired)
    _, pval2 = rank_based_tests(arr1, arr3, paired=paired)
    assert pval1 > 0.1
    assert pval2 < 0.05


@pytest.mark.parametrize(
    "paired, stat_func",
    [(False, _permutation_stat_independent), (True, _permutation_stat_paired)],
)
def test_permutation_stat_functions(permutation_test_arrays, paired, stat_func):
    """Test the permutation test callables for difference in means."""
    arr1, arr2, axis = permutation_test_arrays
    if paired:
        expected = np.mean(arr1 - arr2, axis=axis)
    else:
        expected = np.mean(arr1, axis=axis) - np.mean(arr2, axis=axis)
    test = stat_func(arr1, arr2, axis=axis)
    assert np.allclose(expected, test)


@pytest.mark.parametrize("paired", (False, True))
def test_permutation_test_means_basic(permutation_test_arrays, paired):
    """Test the basic functionality of permutation_test_means."""
    arr1, arr2, axis = permutation_test_arrays
    test = permutation_test_means(arr1, arr2, paired=paired, axis=axis, nres=500, seed=0)
    assert hasattr(test, "statistic")
    assert hasattr(test, "pvalue")
    assert np.all((0 <= test.pvalue) & (test.pvalue <= 1))


@pytest.mark.parametrize("paired", (False, True))
def test_permutation_test_means_reproducibility(permutation_test_arrays, paired):
    """Test reproducibility of permutation_test_means."""
    arr1, arr2, axis = permutation_test_arrays
    test1 = permutation_test_means(
        arr1, arr2, paired=paired, axis=axis, nres=500, seed=42
    )
    test2 = permutation_test_means(
        arr1, arr2, paired=paired, axis=axis, nres=500, seed=42
    )
    assert np.allclose(test1.statistic, test2.statistic)
    assert np.allclose(test1.pvalue, test2.pvalue)
