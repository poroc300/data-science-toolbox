"""Statistical inference functions for hypothesis testing."""

import numpy as np
from scipy.stats import (
    mannwhitneyu,
    permutation_test,
    wilcoxon,
)

from dstools.core_stats import _ttest_ind, _ttest_paired


def t_test(
    arr1,
    arr2,
    paired=False,
    assume_equal_variance=False,
    alternative="two-sided",
    ci=0.95,
    axis=0,
):
    """
    Perform an independent- or paired-samples t-test.

    This function compares the means of two samples under the assumption that
    the data are approximately normally distributed. Depending on the value of
    ``paired``, either an independent-samples t-test or a paired-samples t-test
    is performed. In addition to the hypothesis test, the function returns the
    estimated mean difference, a confidence interval for the difference,
    Cohen's effect size, and the corresponding p-value.

    Parameters
    ----------
    arr1 : array-like
        Array containing observations from the first sample (or first paired
        measurement if ``paired=True``).
    arr2 : array-like
        Array containing observations from the second sample (or second paired
        measurement if ``paired=True``).
    paired : bool, optional
        If False (default), perform an independent-samples t-test. If True,
        perform a paired-samples t-test.
    assume_equal_variances : bool, optional
        Whether to assume that the two populations have equal variances when
        performing an independent-samples t-test. If False (default), perform
        Welch's t-test, which does not assume equal population variances and
        adjusts the degrees of freedom accordingly. If True, perform Student's
        t-test, which pools the sample variances to estimate a common population
        variance. This parameter is ignored if ``paired=True``.
    alternative : {"two-sided", "less", "greater"}, optional
        Defines the alternative hypothesis.
            * "two-sided" (default): the population means are unequal.
            * "less": the mean of sample 1 is less than the mean of sample 2.
            * "greater": the mean of sample 1 is greater than the mean of sample 2.
    ci : float, optional
        Confidence level for the confidence interval of the estimated mean
        difference. Must be between 0 and 1. The default is 0.95, corresponding
        to a 95% confidence interval. The interval is computed using the Student's
        t-distribution. For independent samples, Welch's or Student's t-based
        intervals are used depending on ``assume_equal_variance``. For paired samples,
        the interval is based on the t-distribution of the paired differences.
    axis : int, optional
        Axis corresponding to observations. If ``axis=0`` (default), compute one
        t-test per column. If ``axis=1``, compute one t-test per row.

    Returns
    -------
    dict
        Dictionary containing:
            * "statistic" : t-test statistic.
            * "estimate" : Difference in sample means.
            * "confidence_interval" : Confidence interval for the mean
              difference.
            * "effect_size" : Cohen's d (independent samples) or Cohen's dz
              (paired samples).
            * "pvalue" : p-value of the test.
    """
    # coerce to numpy arrays
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)

    if paired:
        return _ttest_paired(arr1, arr2, alternative, ci, axis)
    else:
        return _ttest_ind(arr1, arr2, assume_equal_variance, alternative, ci, axis)


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
    arr1, arr2 = np.asarray(arr1), np.asarray(arr2)
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
    arr1, arr2 = np.asarray(arr1), np.asarray(arr2)

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
