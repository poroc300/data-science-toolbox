"""Unit tests for functions in _nonparametric.py."""

import itertools

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.stats import mannwhitneyu, wilcoxon

from dstools.core_stats._nonparametric import (
    _mann_whitney,
    _wilcoxon,
    _wilcoxon_rank_sums,
)


@pytest.mark.parametrize("alternative", ("two-sided", "less", "greater"))
def test_mann_whitney_alternatives(non_param_arrays, alternative):
    """Check alternative hypotheses match scipy mannwhitneyu behaviour."""
    arr1 = non_param_arrays["arr1"]
    arr2 = non_param_arrays["arr2"]
    scipy_mw = mannwhitneyu(arr1, arr2, alternative=alternative, axis=0)
    result = _mann_whitney(arr1, arr2, alternative=alternative, axis=0)

    # compare results to expected
    assert_almost_equal(scipy_mw.statistic, result["statistic"], decimal=4)
    assert_almost_equal(scipy_mw.pvalue, result["pvalue"], decimal=4)


def test_mann_whitney_handles_ties():
    """Test the function doesn't break when the samples contain tied values."""
    result = _mann_whitney(
        np.array([1, 2, 2, 3]),
        np.array([2, 3, 3, 4]),
        alternative="two-sided",
        axis=0,
    )
    assert np.isfinite(result["statistic"])
    assert np.isfinite(result["pvalue"])
    assert np.isfinite(result["effect_size"])


@pytest.mark.parametrize("axis, array_length", ((0, 2), (1, 50)))
def test_mann_whitney_axis_support(non_param_arrays, axis, array_length):
    """Check axis argument works as expected."""
    # make 2D arrays
    arr1 = non_param_arrays["arr1"]
    arr2 = non_param_arrays["arr2"]
    arr1 = np.column_stack((arr1, arr1))
    arr2 = np.column_stack((arr2, arr2))

    # output must be two elements
    result = _mann_whitney(
        arr1,
        arr2,
        alternative="two-sided",
        axis=axis,
    )
    assert result["statistic"].size == array_length
    assert result["pvalue"].size == array_length
    assert result["effect_size"].size == array_length


def test_wilcoxon_rank_sums_single_observation():
    """Test function with a single observation."""
    positive, negative = _wilcoxon_rank_sums(np.array([5]), zero_method="wilcox")
    assert positive == 1.0
    assert negative == 0.0


@pytest.mark.parametrize("zero_method", ("wilcox", "pratt", "zsplit"))
def test_wilcoxon_rank_sums_no_zeros_same_results_zero_methods(zero_method):
    """Test that without zeros, all methods should give identical results."""
    diff = np.array([-4, -2, 1, 3])
    positive, negative = _wilcoxon_rank_sums(diff, zero_method=zero_method)
    assert positive == 4
    assert negative == 6


def test_wilcoxon_rank_sums_zero_method_wilcox():
    """Test with wilcox method zeros are excluded before ranking."""
    diff = np.array([-3, -1, 0, 2, 4])
    positive, negative = _wilcoxon_rank_sums(diff, zero_method="wilcox")
    assert positive == 6.0
    assert negative == 4.0


def test_wilcoxon_rank_sums_zero_method_pratt():
    """Test zeros participate in ranking but their ranks are excluded."""
    diff = np.array([-3, -1, 0, 2, 4])
    positive, negative = _wilcoxon_rank_sums(diff, zero_method="pratt")
    assert positive == 8.0
    assert negative == 6.0


def test_wilcoxon_rank_sums_zero_method_zsplit():
    """Test zeros participate in ranking and their ranks are split equally."""
    diff = np.array([-3, -1, 0, 0, 2, 4])
    positive, negative = _wilcoxon_rank_sums(diff, zero_method="zsplit")
    assert positive == 11.5
    assert negative == 9.5


def test_wilcoxon_rank_sums_ties_use_average_ranks():
    """Test that tied absolute differences receive average ranks."""
    diff = np.array([-2, 2, -2, 3])
    positive, negative = _wilcoxon_rank_sums(diff, zero_method="wilcox")
    assert positive == 6
    assert negative == 4


@pytest.mark.parametrize("sign, exp_pos, exp_neg", ((1, 10, 0), (-1, 0, 10)))
def test_wilcoxon_rank_sums_all_positive_negative(sign, exp_pos, exp_neg):
    """Test when differences have the same sign."""
    diff = np.array([1, 2, 3, 4]) * sign
    positive, negative = _wilcoxon_rank_sums(diff, zero_method="wilcox")
    assert exp_pos == positive
    assert exp_neg == negative


@pytest.mark.parametrize(
    "zero_method, expected", (("wilcox", 0), ("pratt", 0), ("zsplit", 5))
)
def test_wilcoxon_rank_sums_all_zeros(zero_method, expected):
    """Test when differences are all zeros. Only zsplit uses zero ranks in sums"""
    diff = np.zeros(4)
    positive, negative = _wilcoxon_rank_sums(diff, zero_method=zero_method)
    assert positive == expected and negative == expected


@pytest.mark.parametrize("axis", (0, 1))
def test_wilcoxon_rank_sums_axis(axis):
    """Test the function supports rank calculation across different axes."""
    diff = np.array([[-1, 4], [2, -5], [3, 6]])
    if axis == 1:
        diff = np.transpose(diff)
    positive, negative = _wilcoxon_rank_sums(diff, zero_method="wilcox", axis=axis)
    assert_array_equal(positive, [5, 4])
    assert_array_equal(negative, [1, 2])


def test_wilcoxon_rank_sums_invalid_zero_method():
    """Invalid zero-handling methods should raise ValueError."""
    with pytest.raises(ValueError, match="zero_method must be one"):
        _wilcoxon_rank_sums(np.ones(3), zero_method="dummy")


@pytest.mark.parametrize(
    "zero_method, alternative",
    itertools.product(("wilcox", "pratt", "zsplit"), ("two-sided", "greater", "less")),
)
def test_wilcoxon_values_match_scipy(non_param_arrays, zero_method, alternative):
    """Check that statistic and p-value match those from scipy."""
    arr1 = non_param_arrays["arr1"]
    arr2 = non_param_arrays["arr2"]
    expected = wilcoxon(arr1, arr2, alternative=alternative, zero_method="wilcox", axis=0)
    result = _wilcoxon(arr1, arr2, alternative, zero_method, axis=0)
    assert result["statistic"] == expected.statistic
    assert result["pvalue"] == expected.pvalue


def test_wilcoxon_keys():
    """Check expected keys are present on the output."""
    result = _wilcoxon(np.arange(1, 5), np.arange(2, 6), "two-sided", "wilcox", axis=0)
    assert set(result.keys()) == {"statistic", "pvalue", "effect_size"}


@pytest.mark.parametrize(
    "arr1, arr2, axis",
    (
        (np.ones(5), np.ones(3), 0),
        (np.ones((3, 3)), np.ones((2, 3)), 0),
        (np.ones((3, 2)), np.ones((2, 3)), 1),
    ),
)
def test_wilcoxon_mismatched_shapes(arr1, arr2, axis):
    """Check a ValueError is raised if arrays don't have the same shape."""
    with pytest.raises(ValueError, match="same number of observations"):
        _wilcoxon(arr1, arr2, "two-sided", "wilcox", axis=axis)


def test_wilcoxon_nan_values():
    """Check a ValueError is raised when missing data is present in any input array."""
    with pytest.raises(ValueError, match="missing data"):
        _wilcoxon(np.array([1, np.nan]), np.ones(2), "two-sided", "wilcox", axis=0)
    with pytest.raises(ValueError, match="missing data"):
        _wilcoxon(np.ones(2), np.array([1, np.nan]), "two-sided", "wilcox", axis=0)


@pytest.mark.parametrize("axis, expected", ((0, 2), (1, 25)))
def test_wilcoxon_axis_result_orientation(non_param_arrays, axis, expected):
    """
    Check output arrays have same length has number of columns when axis=0 and have
    same length as the number of rows when axis=1.
    """
    arr1 = non_param_arrays["arr1"].reshape(25, 2)
    arr2 = non_param_arrays["arr2"].reshape(25, 2)
    result = _wilcoxon(arr1, arr2, "two-sided", "wilcox", axis=axis)
    assert result["statistic"].size == expected
    assert result["pvalue"].size == expected
