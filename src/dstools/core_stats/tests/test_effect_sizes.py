"""Unit tests for functions in effect_sizes.py."""

import numpy as np
import pingouin as pg
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from scipy.stats import ttest_rel

from dstools.core_stats import _cohens_d_independent, _cohens_dz_paired


@pytest.fixture(scope="module")
def sample_arrays():
    """Sample arrays for testing effect size calculations."""
    arr1 = np.arange(1, 6)
    arr1_2d = np.column_stack((np.arange(1, 5), np.arange(2, 6)))
    return {
        "arr1_1d": arr1,
        "arr2_1d": arr1 + np.array([1, 2, 2, 1, 4]),
        "arr1_2d": arr1_2d,
        "arr2_2d": arr1_2d + np.repeat([1, 2], repeats=2).reshape(-1, 1),
    }


def test_cohens_d_independent_1d(sample_arrays):
    """Cohen's d matches the calculation from the package pinguoin."""
    arr1 = sample_arrays["arr1_1d"]
    arr2 = sample_arrays["arr2_1d"]
    expected = pg.compute_effsize(arr1, arr2, eftype="cohen", paired=False)
    result = _cohens_d_independent(arr1, arr2)
    assert_almost_equal(expected, result, decimal=3)


@pytest.mark.parametrize("axis", (0, 1))
def test_cohens_d_independent_axis(sample_arrays, axis):
    """Computes independent Cohen's d using the axis argument for 2D arrays."""
    arr1 = sample_arrays["arr1_2d"]
    arr2 = sample_arrays["arr2_2d"]

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


def test_cohens_dz_paired_1d(sample_arrays):
    """Cohen's dz paired matches the calculation from the package pinguoin."""
    arr1 = sample_arrays["arr1_1d"]
    arr2 = sample_arrays["arr2_1d"]
    t_stat, _ = ttest_rel(arr1, arr2)
    expected = t_stat / np.sqrt(arr1.size)
    result = _cohens_dz_paired(arr1, arr2)
    assert_almost_equal(expected, result, decimal=3)


@pytest.mark.parametrize("axis", (0, 1))
def test_cohens_dz_paired_axis(sample_arrays, axis):
    """
    Cohen's dz paired matches the equivalent effect size computed from the
    paired-samples t-statistic (dz = t / sqrt(n)).
    """
    arr1 = sample_arrays["arr1_2d"]
    arr2 = sample_arrays["arr2_2d"]

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
