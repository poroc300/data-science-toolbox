"""Unit tests for functions in _nonparametric.py."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.stats import mannwhitneyu

from dstools.core_stats._nonparametric import _mann_whitney, _wilcoxon_rank_sums


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


@pytest.mark.parametrize("zero_method", ["wilcox", "pratt", "zsplit"])
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

##############################################################################
##############################################################################
##############################################################################



def test_all_positive():
    diff = np.array([1, 2, 3, 4])

    positive, negative = _wilcoxon_rank_sums(diff)

    assert positive == 10.0
    assert negative == 0.0


def test_all_negative():
    diff = np.array([-1, -2, -3, -4])

    positive, negative = _wilcoxon_rank_sums(diff)

    assert positive == 0.0
    assert negative == 10.0


def test_all_zeros_wilcox():
    """All zeros are removed, leaving no ranks."""
    diff = np.zeros(4)

    positive, negative = _wilcoxon_rank_sums(
        diff,
        zero_method="wilcox",
    )

    assert positive == 0.0
    assert negative == 0.0


def test_all_zeros_pratt():
    """All zeros are ranked but excluded from both sums."""
    diff = np.zeros(4)

    positive, negative = _wilcoxon_rank_sums(
        diff,
        zero_method="pratt",
    )

    assert positive == 0.0
    assert negative == 0.0


def test_all_zeros_zsplit():
    """All zero ranks are split equally between positive and negative sums."""
    diff = np.zeros(4)

    positive, negative = _wilcoxon_rank_sums(
        diff,
        zero_method="zsplit",
    )

    # All four observations are tied at rank 2.5.
    # Total zero rank sum = 10, split equally.
    assert positive == 5.0
    assert negative == 5.0


def test_axis_1():
    """The function should support paired observations along axis=1."""
    diff = np.array(
        [
            [-1, 2, 3],
            [4, -5, 6],
        ]
    )

    positive, negative = _wilcoxon_rank_sums(
        diff,
        zero_method="wilcox",
        axis=1,
    )

    # Row 1: abs = [1, 2, 3] -> ranks [1, 2, 3]
    # W+ = 5, W- = 1
    #
    # Row 2: abs = [4, 5, 6] -> ranks [1, 2, 3]
    # W+ = 4, W- = 2
    assert_allclose(positive, [5.0, 4.0])
    assert_allclose(negative, [1.0, 2.0])


def test_2d_input_axis_0():
    """Each column should be ranked independently."""
    diff = np.array(
        [
            [-1, 4],
            [2, -5],
            [3, 6],
        ]
    )

    positive, negative = _wilcoxon_rank_sums(diff)

    # Column 1: [-1, 2, 3] -> ranks [1, 2, 3]
    # W+ = 5, W- = 1
    #
    # Column 2: [4, -5, 6] -> ranks [1, 2, 3]
    # W+ = 4, W- = 2
    assert_allclose(positive, [5.0, 4.0])
    assert_allclose(negative, [1.0, 2.0])


def test_2d_input_with_zeros():
    """Zero handling should happen independently for each test."""
    diff = np.array(
        [
            [-1, 0],
            [2, 2],
            [0, -3],
            [4, 4],
        ]
    )

    positive, negative = _wilcoxon_rank_sums(
        diff,
        zero_method="zsplit",
    )

    # Column 1:
    # values [-1, 2, 0, 4]
    # abs     [1, 2, 0, 4]
    # ranks   [2, 3, 1, 4]
    # W+ = 3 + 4 + 0.5 = 7.5
    # W- = 2 + 0.5 = 2.5
    #
    # Column 2:
    # values [0, 2, -3, 4]
    # abs     [0, 2, 3, 4]
    # ranks   [1, 2, 3, 4]
    # W+ = 2 + 4 + 0.5 = 6.5
    # W- = 3 + 0.5 = 3.5
    assert_allclose(positive, [7.5, 6.5])
    assert_allclose(negative, [2.5, 3.5])


@pytest.mark.parametrize(
    "zero_method",
    ["invalid", "", None, 1],
)
def test_invalid_zero_method(zero_method):
    """Invalid zero-handling methods should raise ValueError."""
    with pytest.raises(
        ValueError,
        match="zero_method must be one of 'wilcox', 'pratt', or 'zsplit'",
    ):
        _wilcoxon_rank_sums(
            np.array([-1, 0, 1]),
            zero_method=zero_method,
        )


def test_single_observation():
    diff = np.array([5])

    positive, negative = _wilcoxon_rank_sums(diff)

    assert positive == 1.0
    assert negative == 0.0


def test_negative_single_observation():
    diff = np.array([-5])

    positive, negative = _wilcoxon_rank_sums(diff)

    assert positive == 0.0
    assert negative == 1.0


def test_return_types_for_1d_input():
    diff = np.array([-2, 1, 3])

    positive, negative = _wilcoxon_rank_sums(diff)

    assert isinstance(positive, np.float64)
    assert isinstance(negative, np.float64)


def test_return_shapes_for_2d_input():
    diff = np.array(
        [
            [-1, 2],
            [3, -4],
            [5, 6],
        ]
    )

    positive, negative = _wilcoxon_rank_sums(diff, axis=0)

    assert positive.shape == (2,)
    assert negative.shape == (2,)


def test_nan_with_wilcox():
    """NaNs should not contribute to the rank sums."""
    diff = np.array([-1, 2, np.nan, 4])

    positive, negative = _wilcoxon_rank_sums(
        diff,
        zero_method="wilcox",
    )

    # Non-NaN absolute values: 1, 2, 4 -> ranks 1, 2, 3
    assert positive == 5.0
    assert negative == 1.0
