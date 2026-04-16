"""Unit tests for functions in inference.py."""

import numpy as np
import pytest

from dscience_utils.core_stats import t_test

np.random.seed(7)


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
