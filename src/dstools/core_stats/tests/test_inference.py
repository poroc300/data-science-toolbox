"""Unit tests for functions in inference.py."""

import itertools

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel, wilcoxon
from scipy.stats import permutation_test as scipy_perm_test

from dstools.core_stats import permutation_test, rank_test, t_test


@pytest.mark.parametrize("paired", (True, False))
def test_t_test(ttest_arrays, paired):
    """Test t_test with different values for ``paired`` argument."""
    arr1 = ttest_arrays["arr1"]
    arr2 = ttest_arrays["arr2"]

    # get results from scipy functions
    if paired:
        expected = ttest_rel(arr1, arr2)
    else:
        expected = ttest_ind(arr1, arr2, equal_var=False)

    # get results from t_test and make assertions
    result = t_test(arr1, arr2, paired=paired)
    assert_almost_equal(expected.statistic, result["statistic"], decimal=4)


@pytest.mark.parametrize("paired", (True, False))
def test_rank_test(non_param_arrays, paired):
    """Test t_test with different values for ``paired`` argument."""
    arr1 = non_param_arrays["arr1"]
    arr2 = non_param_arrays["arr2"]

    # get results from scipy functions
    if paired:
        expected = wilcoxon(arr1, arr2, zero_method="wilcox")
    else:
        expected = mannwhitneyu(arr1, arr2)

    # get results from t_test and make assertions
    result = rank_test(arr1, arr2, paired=paired)
    assert_almost_equal(expected.statistic, result["statistic"], decimal=4)


def test_permutation_test_is_reproducible_with_seed(non_param_arrays):
    """Test function yields same results with same seed."""
    arr1 = non_param_arrays["arr1"]
    arr2 = non_param_arrays["arr2"]
    result1 = permutation_test(arr1, arr2, nres=50, random_state=10)
    result2 = permutation_test(arr1, arr2, nres=50, random_state=10)
    assert result1["statistic"] == result2["statistic"]
    assert result1["pvalue"] == result2["pvalue"]


@pytest.mark.parametrize(
    "paired, alternative, axis",
    itertools.product((True, False), ("two-sided", "greater", "less"), (0, 1)),
)
def test_permutation_test_matches_scipy(non_param_arrays, paired, alternative, axis):
    """Test scipy version of permutation test is equivalent."""
    arr1 = non_param_arrays["arr1"].reshape(25, 2)
    arr2 = non_param_arrays["arr2"].reshape(25, 2)
    _type = "samples" if paired else "independent"
    stat_func = lambda x, y: np.mean(x) - np.mean(y)

    # run both functions
    expected = scipy_perm_test(
        (arr1, arr2),
        statistic=stat_func,
        n_resamples=50,
        alternative=alternative,
        permutation_type=_type,
        random_state=10,
        axis=axis,
    )
    result = permutation_test(
        arr1,
        arr2,
        nres=50,
        alternative=alternative,
        paired=paired,
        axis=axis,
        random_state=10,
    )

    # assertions
    assert_almost_equal(expected.statistic, result["statistic"], decimal=3)
    assert_almost_equal(expected.pvalue, result["pvalue"], decimal=3)


def test_permutation_test_supports_custom_statistic():
    """Test a custom statistic can be supplied."""
    result = permutation_test(
        [1, 2, 100],
        [2, 3, 4],
        statistic=np.median,
        nres=np.inf,
    )
    assert_almost_equal(result["statistic"], -1.0)


def test_permutation_test_supports_statistic_kwargs():
    """Test that kwargs can be supplied to statistic."""
    result = permutation_test(
        [1, 2, 3, 100],
        [2, 3, 4, 5],
        statistic=np.mean,
        statistic_kwargs={"dtype": float},
        nres=50,
        random_state=10,
    )
    assert isinstance(result["statistic"], float)


def test_permutation_test_mismatched_shapes():
    """Test function raises ValueError when paired=True and input arrays have different shapes."""
    with pytest.raises(ValueError, match="for paired test, arr1 and arr2"):
        permutation_test([1, 2, 3], [1, 2], paired=True)
