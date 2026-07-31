"""Unit tests for functions in inference.py."""

import numpy as np
import pingouin as pg
import pytest
from numpy.testing import assert_allclose, assert_almost_equal
from scipy.stats import ttest_ind, ttest_rel

from dstools.core_stats._effect_sizes import _cohens_dz_paired
from dstools.core_stats._parametric import _ttest_ind, _ttest_paired


def test_ttest_ind_keys(ttest_arrays):
    """Tests expected keys are returned."""
    result = _ttest_ind(
        ttest_arrays["arr1"],
        ttest_arrays["arr2"],
        assume_equal_variance=False,
        alternative="two-sided",
        ci=0.95,
        axis=0,
    )
    exp_keys = {"statistic", "estimate", "ci", "effect_size", "pvalue"}
    assert exp_keys == set(result.keys())


def test_ttest_ind_output(ttest_arrays):
    """Test if output of _ttest_ind is correct."""
    arr1 = ttest_arrays["arr1"]
    arr2 = ttest_arrays["arr2"]

    # get expected output
    scipy_ttest = ttest_ind(
        arr1,
        arr2,
        equal_var=False,
        alternative="two-sided",
        axis=0,
    )
    stat = scipy_ttest.statistic
    pval = scipy_ttest.pvalue
    ci_level = scipy_ttest.confidence_interval(confidence_level=0.95)
    ci = (ci_level.low, ci_level.high)
    estimate = np.mean(arr1) - np.mean(arr2)
    efsize = pg.compute_effsize(arr1, arr2, eftype="cohen", paired=False)

    # compare results to expected
    result = _ttest_ind(
        arr1,
        arr2,
        assume_equal_variance=False,
        alternative="two-sided",
        ci=0.95,
        axis=0,
    )
    assert_almost_equal(stat, result["statistic"], decimal=4)
    assert_almost_equal(pval, result["pvalue"], decimal=4)
    assert_allclose(ci, result["ci"], atol=0.001)
    assert_almost_equal(estimate, result["estimate"], decimal=4)
    assert_almost_equal(efsize, result["effect_size"], decimal=4)


@pytest.mark.parametrize("alternative", ("two-sided", "less", "greater"))
def test_ttest_ind_alternatives(ttest_arrays, alternative):
    """Check alternative hypotheses match scipy ttest_ind behaviour."""
    arr1 = ttest_arrays["arr1"]
    arr2 = ttest_arrays["arr2"]

    scipy_ttest = ttest_ind(
        arr1,
        arr2,
        equal_var=False,
        alternative=alternative,
        axis=0,
    )

    # compare results to expected
    result = _ttest_ind(
        arr1,
        arr2,
        assume_equal_variance=False,
        alternative=alternative,
        ci=0.95,
        axis=0,
    )
    assert_almost_equal(scipy_ttest.statistic, result["statistic"], decimal=4)
    assert_almost_equal(scipy_ttest.pvalue, result["pvalue"], decimal=4)


def test_ttest_paired_keys(ttest_arrays):
    """Tests expected keys are returned."""
    result = _ttest_paired(
        ttest_arrays["arr1"],
        ttest_arrays["arr2"],
        alternative="two-sided",
        ci=0.95,
        axis=0,
    )
    exp_keys = {"statistic", "estimate", "ci", "effect_size", "pvalue"}
    assert exp_keys == set(result.keys())


def test_ttest_paired_output(ttest_arrays):
    """Test if output of _ttest_paired is correct."""
    arr1 = ttest_arrays["arr1"]
    arr2 = ttest_arrays["arr2"]

    # get expected output
    scipy_ttest = ttest_rel(
        arr1,
        arr2,
        alternative="two-sided",
        axis=0,
    )
    stat = scipy_ttest.statistic
    pval = scipy_ttest.pvalue
    ci_level = scipy_ttest.confidence_interval(confidence_level=0.95)
    ci = (ci_level.low, ci_level.high)
    estimate = np.mean(arr1) - np.mean(arr2)
    efsize = _cohens_dz_paired(arr1, arr2, axis=0)

    # compare results to expected
    result = _ttest_paired(
        arr1,
        arr2,
        alternative="two-sided",
        ci=0.95,
        axis=0,
    )
    assert_almost_equal(stat, result["statistic"], decimal=4)
    assert_almost_equal(pval, result["pvalue"], decimal=4)
    assert_allclose(ci, result["ci"], atol=0.001)
    assert_almost_equal(estimate, result["estimate"], decimal=4)
    assert_almost_equal(efsize, result["effect_size"], decimal=4)


@pytest.mark.parametrize("alternative", ("two-sided", "less", "greater"))
def test_ttest_paired_alternatives(ttest_arrays, alternative):
    """Check alternative hypotheses match scipy ttest_rel behaviour."""
    arr1 = ttest_arrays["arr1"]
    arr2 = ttest_arrays["arr2"]

    scipy_ttest = ttest_rel(arr1, arr2, alternative=alternative, axis=0)

    # compare results to expected
    result = _ttest_paired(
        arr1,
        arr2,
        alternative=alternative,
        ci=0.95,
        axis=0,
    )
    assert_almost_equal(scipy_ttest.statistic, result["statistic"], decimal=4)
    assert_almost_equal(scipy_ttest.pvalue, result["pvalue"], decimal=4)


def test_ttest_paired_arrays_equal_lengths():
    """Check paired length validation."""
    with pytest.raises(ValueError, match="same number of observations"):
        _ttest_paired(
            np.ones(3),
            np.ones(2),
            alternative="two-sided",
            ci=0.95,
            axis=0,
        )


@pytest.mark.parametrize("axis, array_length", ((0, 2), (1, 50)))
def test_ttest_ind_axis_support(ttest_arrays, axis, array_length):
    """Check independent t-test axis support."""
    # make 2D arrays
    arr1 = ttest_arrays["arr1"]
    arr2 = ttest_arrays["arr2"]
    arr1 = np.column_stack((arr1, arr1))
    arr2 = np.column_stack((arr2, arr2))

    # output must be two elements
    result = _ttest_ind(
        arr1,
        arr2,
        assume_equal_variance=False,
        alternative="two-sided",
        ci=0.95,
        axis=axis,
    )
    assert result["statistic"].size == array_length
    assert result["pvalue"].size == array_length
