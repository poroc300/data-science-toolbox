"""Unit tests for functions in effect_sizes.py."""

import numpy as np
import pingouin as pg
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from scipy.stats import ttest_rel

from dstools.core_stats._effect_sizes import _cohens_d_independent, _cohens_dz_paired


def test_cohens_d_independent_1d(efsize_arrays):
    """Cohen's d matches the calculation from the package pinguoin."""
    arr1 = efsize_arrays["arr1_1d"]
    arr2 = efsize_arrays["arr2_1d"]
    expected = pg.compute_effsize(arr1, arr2, eftype="cohen", paired=False)
    result = _cohens_d_independent(arr1, arr2)
    assert_almost_equal(expected, result, decimal=3)


@pytest.mark.parametrize("axis", (0, 1))
def test_cohens_d_independent_axis(efsize_arrays, axis):
    """Computes independent Cohen's d using the axis argument for 2D arrays."""
    arr1 = efsize_arrays["arr1_2d"]
    arr2 = efsize_arrays["arr2_2d"]

    # column-wise expected result
    if axis == 0:
        expected = np.array(
            [
                _cohens_d_independent(arr1[:, 0], arr2[:, 0]),
                _cohens_d_independent(arr1[:, 1], arr2[:, 1]),
            ]
        )
    # row-wise expected result
    else:
        expected = np.array(
            [_cohens_d_independent(arr1[i], arr2[i]) for i in range(arr1.shape[0])]
        )

    # get result and make assertion
    result = _cohens_d_independent(arr1, arr2, axis=axis)
    assert_allclose(expected, result)


def test_cohens_dz_paired_1d(efsize_arrays):
    """Cohen's dz paired matches the calculation from the package pinguoin."""
    arr1 = efsize_arrays["arr1_1d"]
    arr2 = efsize_arrays["arr2_1d"]
    t_stat, _ = ttest_rel(arr1, arr2)
    expected = t_stat / np.sqrt(arr1.size)
    result = _cohens_dz_paired(arr1, arr2)
    assert_almost_equal(expected, result, decimal=3)


@pytest.mark.parametrize("axis", (0, 1))
def test_cohens_dz_paired_axis(efsize_arrays, axis):
    """
    Cohen's dz paired matches the equivalent effect size computed from the
    paired-samples t-statistic (dz = t / sqrt(n)).
    """
    arr1 = efsize_arrays["arr1_2d"]
    arr2 = efsize_arrays["arr2_2d"]

    # column-wise expected result
    if axis == 0:
        expected = np.array(
            [
                _cohens_dz_paired(arr1[:, 0], arr2[:, 0]),
                _cohens_dz_paired(arr1[:, 1], arr2[:, 1]),
            ]
        )
    # row-wise expected result
    else:
        expected = np.array(
            [_cohens_dz_paired(arr1[i], arr2[i]) for i in range(arr1.shape[0])]
        )

    # get result and make assertion
    result = _cohens_dz_paired(arr1, arr2, axis=axis)
    assert_allclose(expected, result)
