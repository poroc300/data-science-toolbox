"""Unit tests for functions in inference.py."""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy.stats import ttest_ind, ttest_rel

from dstools.core_stats import t_test


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
