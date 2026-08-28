"""Unit tests for functions in _nonparametric.py."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.stats import mannwhitneyu

from dstools.core_stats._nonparametric import _mann_whitney


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
