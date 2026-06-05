"""Statistical inference functions for hypothesis testing."""

import numpy as np
from scipy.stats import (
    mannwhitneyu,
    permutation_test,
    ttest_ind,
    ttest_rel,
    wilcoxon,
)

from dscience_toolbox.utils import _to_numpy


def t_test(arr1, arr2, paired=False, alternative="two-sided", **kwargs):
    """
    Perform a t-test (independent or paired) of difference of means between two
    samples. This function wraps scipy.stats.ttest_ind and scipy.stats.ttest_rel.
    To be used when samples are normally distributed.

    Parameters
    ----------
    arr1 : array-like
        Array of the first sample (before sample in case of paired=True).
    arr2 : array-like
        Array of the second sample (after sample in case of paired=True).
    paired : bool, optional
        Perform an independent t-test if False (default) or a paired t-test if
        True.
    alternative : str, optional
        Defines the alternative hypothesis. The available options are:
            * "two-sided" - the means of the sample distributions are unequal (default).
            * "less" - the mean of sample 1 is lower than the mean of sample 2.
            * "greater" - the mean of sample 1 is higher than the mean of sample 2.
    **kwargs : dict, optional
       Additional keyword arguments passed to scipy.stats.ttest_ind if paired=False
       or to scipy.stats.ttest_rel in case paired=True.

    Returns
    -------
    TtestResult
        Class which contains the "statistic" and "pvalue" attributes.
    """
    arr1, arr2 = _to_numpy(arr1), _to_numpy(arr2)
    if paired:
        if arr1.size != arr2.size:
            msg = "both samples must have the same length in a paired t-test"
            raise ValueError(msg)
        return ttest_rel(arr1, arr2, alternative=alternative, **kwargs)
    return ttest_ind(arr1, arr2, alternative=alternative, **kwargs)


def rank_based_tests(arr1, arr2, paired=False, alternative="two-sided", **kwargs):
    """
    Perform a rank-based nonparametric comparison between two samples. This function
    wraps scipy.stats.mannwhitneyu and scipy.stats.wilcoxon.

    Parameters
    ----------
    arr1 : array-like
        First sample of observations.
    arr2 : array-like
        Second sample of observations.
    paired : bool, optional
        Whether the two samples are paired (dependent observations). If True,
        the Wilcoxon signed-rank test is used. If False, the Mann-Whitney U test
        is used. Default is False.
    alternative : str, optional
        Defines the alternative hypothesis. The available options are:
            * "two-sided" - the means of the sample distributions are unequal (default).
            * "less" - the mean of sample 1 is lower than the mean of sample 2.
            * "greater" - the mean of sample 1 is higher than the mean of sample 2.
    **kwargs : dict, optional
       Additional keyword arguments passed to scipy.stats.mannwhitneyu if paired=False
       or to scipy.stats.wilcoxon in case paired=True.

    Returns
    -------
    WilcoxonResult or MannwhitneyuResult
        Classes which contain the "statistic" and "pvalue" attributes.
    """
    arr1, arr2 = _to_numpy(arr1), _to_numpy(arr2)
    if paired:
        if arr1.size != arr2.size:
            msg = "both samples must have the same length in a Wilcoxon test"
            raise ValueError(msg)
        return wilcoxon(arr1, arr2, alternative=alternative, **kwargs)
    return mannwhitneyu(arr1, arr2, alternative=alternative, **kwargs)


def _permutation_stat_independent(arr1, arr2, axis):
    """
    Callable to be used by the permutation_test function. This callable used the
    difference in observed means.

    Parameters
    ----------
    arr1 : array-like
        First sample of observations.
    arr2 : array-like
        Second sample of observations.  Must be broadcastable to the same
        shape as arr1.
    axis : int
        Axis along which to compute the mean difference. For 2D arrays:
        - axis=0 computes differences column-wise (rows are observations)
        - axis=1 computes differences row-wise (columns are observations)

    Returns
    -------
    ndarray or numpy scalar
        Difference between the means of both samples.
    """
    return np.mean(arr1, axis=axis) - np.mean(arr2, axis=axis)


def _permutation_stat_paired(arr1, arr2, axis):
    """
    Callable to be used by the permutation_test function. This callable used the
    difference in observed means.

    Parameters
    ----------
    arr1 : array-like
        First set of paired observations.
    arr2 : array-like
        Second set of paired observations. Must be broadcastable to the same
        shape as arr1.
    axis : int
        Axis along which to compute the mean difference. For 2D arrays:
        - axis=0 computes differences column-wise (rows are observations)
        - axis=1 computes differences row-wise (columns are observations)

    Returns
    -------
    ndarray or numpy scalar
        Mean of the pairwise differences along the specified axis.
    """
    return np.mean(arr1 - arr2, axis=axis)


def permutation_test_means(
    arr1, arr2, paired=False, nres=10_000, alternative="two-sided", axis=0, seed=10
):
    """
    Perform a permutation test to test if the means between two samples are significantly
    different. This function wraps scipy.stats.permutation_test.

    Parameters
    ----------
    arr1 : array-like
        First sample of observations.
    arr2 : array-like
        Second sample of observations.
    paired : bool, optional
        If True, perform a paired permutation test using within-pair sign
        flipping. In this case, arr1 and arr2 must have the same shape. If False
        (default), perform an independent two-sample permutation test using label
        shuffling.
    nres : int or np.inf, optional
        Number of random permutations (resamples) used to approximate the null
        distribution. If greater than or equal to the number of distinct permutations,
        the exact null distribution will be computed. The default is 10,000 resamples.
    alternative : str, optional
        Defines the alternative hypothesis:
         - "two-sided": means are different (default)
         - "less": mean(arr1) < mean(arr2)
         - "greater": mean(arr1) > mean(arr2)
    axis : int, optional
        The axis of the (broadcasted) samples over which to calculate the
        difference in observed means in the case of 2D arrays. If axis=0, differences
        are calculated column-wise, while row-wise mean differences will be computed
        if axis=1. The default is 0.
    seed : int or None, optional
        Seed for the random number generator passed to scipy. Default is 10.

    Returns
    -------
    result : PermutationTestResult
        Object containing:
       - ``statistic`` : observed test statistic
       - ``pvalue`` : p-value of the test
       - ``null_distribution`` : array of resampled statistics (if available)
    """
    arr1, arr2 = _to_numpy(arr1), _to_numpy(arr2)

    # permutation test for paired sample statistics
    if paired:
        if arr1.shape != arr2.shape:
            raise ValueError("for paired test, arr1 and arr2 must have the same shape")
        stat_func = _permutation_stat_paired
        permutation_type = "samples"

    # permutation test for independent samples
    else:
        stat_func = _permutation_stat_independent
        permutation_type = "independent"

    # perform permutation test
    result = permutation_test(
        (arr1, arr2),
        statistic=stat_func,
        permutation_type=permutation_type,
        n_resamples=nres,
        alternative=alternative,
        random_state=seed,
        axis=axis,
    )
    return result
