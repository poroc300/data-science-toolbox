"""Private functions to apply non-parametric statistical tests."""

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
from scipy import stats

from dstools.core_stats._effect_sizes import _rank_biserial_correlation_independent


def _mann_whitney(arr1, arr2, alternative, axis):
    """
    Perform a non-parametric Mann-Whitney U test for two independent samples.
    The test evaluates whether observations from one sample tend to be systematically
    larger or smaller than observations from the other sample.

    Parameters
    ----------
    arr1 : ndarray
        Array containing observations from the first sample.
    arr2 : ndarray
        Array containing observations from the second sample.
    alternative : {"two-sided", "less", "greater"}
        Defines the alternative hypothesis.

        * ``"two-sided"``: the distributions of the two samples differ.
        * ``"less"``: the first sample tends to have smaller values than the
          second sample.
        * ``"greater"``: the first sample tends to have larger values than the
          second sample.
    axis : int
        Axis corresponding to observations. If ``axis=0``, compute one test
        per column. If ``axis=1``, compute one test per row.

    Returns
    -------
    dict
        Dictionary containing:

        * "statistic" : Mann-Whitney U test statistic.
        * "pvalue" : p-value corresponding to the Mann-Whitney U test.
        * "effect_size" : Rank-biserial correlation measuring the direction and magnitude of
          the difference between the two independent samples.

    Notes
    -----
    The test is performed using SciPy's ``mannwhitneyu`` implementation,
    including its handling of ties and other rank-related edge cases.
    """
    # calculate U statistics
    result = mannwhitneyu(
        arr1,
        arr2,
        alternative=alternative,
        axis=axis,
    )

    # calculate rank biserial correlation for independent samples
    n1 = arr1.shape[axis]
    n2 = arr2.shape[axis]
    effect_size = _rank_biserial_correlation_independent(
        result.statistic,
        n1,
        n2,
    )
    return {
        "statistic": result.statistic,
        "effect_size": effect_size,
        "pvalue": result.pvalue,
    }


def _wilcoxon(arr1, arr2, alternative, zero_method, axis):
    """Perform a Wilcoxon signed-rank test."""
    if arr1.shape[axis] != arr2.shape[axis]:
        raise ValueError(
            "both samples must have the same number of observations "
            "along the specified axis in a paired test"
        )

    result = wilcoxon(
        arr1,
        arr2,
        alternative=alternative,
        zero_method=zero_method,
        axis=axis,
    )

    # Calculate the rank-biserial correlation from the paired differences.
    diff = arr1 - arr2

    effect_size = _wilcoxon_rank_biserial(
        diff,
        zero_method=zero_method,
        axis=axis,
    )

    return {
        "statistic": result.statistic,
        "effect_size": effect_size,
        "pvalue": result.pvalue,
    }
