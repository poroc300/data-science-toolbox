"""Unit tests for functions in inference.py."""

import numpy as np
import pytest

from dscience_toolbox.core_stats import (
    _permutation_stat_independent,
    _permutation_stat_paired,
    permutation_test_means,
    rank_based_tests,
    t_test,
)


@pytest.mark.parametrize("paired", (True, False))
def test_t_test(paired):
    """Test independent and paired t-test statistics."""
    # create arrays for testing
    arr1 = np.arange(50)
    arr2 = arr1 + np.tile([-1, 1], 25)
    arr3 = arr1 + np.random.default_rng(0).normal(100, 5, size=50)

    # assertions
    _, pval1 = t_test(arr1, arr2, paired=paired)
    _, pval2 = t_test(arr1, arr3, paired=paired)
    assert pval1 > 0.1
    assert pval2 < 0.05


def test_t_test_paired_length_mismatch():
    """Test error is raised when input arrays have different shape."""
    with pytest.raises(
        ValueError,
        match="must have the same length",
    ):
        t_test(np.arange(5), np.arange(4), paired=True)


@pytest.mark.parametrize("alternative", ("less", "greater"))
def test_t_test_alternative(alternative):
    """Test alternative argument."""
    arr1 = np.ones(5)
    arr2 = np.full((5,), 10)
    result = t_test(arr1, arr2, alternative=alternative)
    if alternative == "less":
        assert result.pvalue < 0.05
    else:
        assert result.pvalue > 0.95


@pytest.mark.parametrize("paired", (True, False))
def test_rank_based_tests(paired):
    """Test independent and paired rank-based statistics."""
    # create arrays for testing
    arr1 = np.arange(50)
    arr2 = arr1 + np.tile([-1, 1], 25)
    arr3 = arr1 + np.random.default_rng(0).normal(100, 5, size=50)

    # assertions
    _, pval1 = rank_based_tests(arr1, arr2, paired=paired)
    _, pval2 = rank_based_tests(arr1, arr3, paired=paired)
    assert pval1 > 0.1
    assert pval2 < 0.05


def test_rank_based_tests_paired_length_mismatch():
    """Test error is raised when input arrays have different shape for Wilcoxon test."""
    with pytest.raises(ValueError, match="must have the same length"):
        rank_based_tests(np.arange(5), np.arange(4), paired=True)


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


def test_permutation_test_means_paired_shape_mismatch():
    """Test error is raised when input arrays have different shapes."""
    with pytest.raises(ValueError, match="same shape"):
        permutation_test_means(np.ones((10, 2)), np.ones((10, 3)), paired=True)


@pytest.mark.parametrize("axis", (0, 1))
def test_permutation_axis_shape(axis):
    """Test if differences are calculated column-wise and row-wise."""
    result = permutation_test_means(
        np.ones((20, 3)),
        np.zeros((20, 3)),
        axis=axis,
        nres=100,
    )
    if axis == 0:
        assert result.statistic.shape == (3,)
    else:
        assert result.statistic.shape == (20,)
