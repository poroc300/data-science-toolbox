"""Unit tests for functions in inference.py."""

import pytest
from numpy.testing import assert_almost_equal
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel, wilcoxon

from dstools.core_stats import rank_test, t_test


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
