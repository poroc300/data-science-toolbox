"""Private functions to apply nonparametric statistical tests."""

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

from dstools.core_stats._effect_sizes import _cohens_d_independent, _cohens_dz_paired


def _mannwhitneyu(arr1, arr2, alternative, axis):
    """
    Perform a Mann-Whitney U test. This function compares the distributions of two 
    independent samples without assuming normality.

    Parameters
    ----------
    arr1 : ndarray
        Array containing observations from the first sample.
    arr2 : ndarray
        Array containing observations from the second sample.
    alternative : {"two-sided", "less", "greater"}
        Defines the alternative hypothesis.
            * "two-sided": the distributions of the two samples are unequal.
            * "less": the distribution of sample 1 is stochastically less than sample 2.
            * "greater": the distribution of sample 1 is stochastically greater than sample 2.
    axis : int
        Axis corresponding to observations. If ``axis=0``, compute one
        Mann-Whitney U test per column. If ``axis=1``, compute one test per row.

    Returns
    -------
    dict
        Dictionary containing:
            * "statistic" : Mann-Whitney U test statistic.
            * "estimate" : Difference in sample medians.
            * "effect_size" : Cohen's d (independent samples)
            * "pvalue" : p-value of the test.
    """
    # observed difference between sample medians
    estimate = np.median(arr1, axis=axis) - np.median(arr2, axis=axis)

    # perform Mann-Whitney U test
    result = mannwhitneyu(
        arr1,
        arr2,
        alternative=alternative,
        axis=axis,
    )

    # calculate effect size and return output as dict
    effect_size = _cohens_d_independent(arr1, arr2, axis=axis)
    return {
        "statistic": result.statistic,
        "estimate": estimate,
        "effect_size": effect_size,
        "pvalue": result.pvalue,
    }


def _wilcoxon(arr1, arr2, alternative, axis):
    """
    Perform a Wilcoxon signed-rank test. This function compares the distributions 
    of two paired samples without assuming normality.

    Parameters
    ----------
    arr1 : ndarray
        Array containing observations from the first paired measurement.
    arr2 : ndarray
        Array containing observations from the second paired measurement.
    alternative : {"two-sided", "less", "greater"}
        Defines the alternative hypothesis.
            * "two-sided": the distributions of the two samples are unequal.
            * "less": the distribution of sample 1 is stochastically less than sample 2.
            * "greater": the distribution of sample 1 is stochastically greater than sample 2.
    axis : int
        Axis corresponding to observations. If ``axis=0``, compute one
        Wilcoxon test per column. If ``axis=1``, compute one test per row.

    Returns
    -------
    dict
        Dictionary containing:
            * "statistic" : Wilcoxon signed-rank test statistic.
            * "estimate" : Difference in sample medians.
            * "effect_size" : Cohen's dz (paired samples).
            * "pvalue" : p-value of the test.
    """
    # observed difference between sample medians
    estimate = np.median(arr1, axis=axis) - np.median(arr2, axis=axis)

    # samples must have same length
    if arr1.shape[axis] != arr2.shape[axis]:
        raise ValueError(
            "both samples must have the same number of observations "
            "along the specified axis in a paired Wilcoxon test"
        )

    # perform Wilcoxon signed-rank test
    result = wilcoxon(
        arr1,
        arr2,
        alternative=alternative,
        axis=axis,
    )

    # calculate Cohen's dz and return output as dict
    effect_size = _cohens_dz_paired(arr1, arr2, axis=axis)
    return {
        "statistic": result.statistic,
        "estimate": estimate,
        "effect_size": effect_size,
        "pvalue": result.pvalue,
    }
