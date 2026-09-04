"""Statistical inference functions for hypothesis testing."""

import numpy as np
from scipy.stats import permutation_test

from dstools.core_stats._parametric import _ttest_ind, _ttest_paired
from dstools.core_stats._nonparametric import _mann_whitney, _wilcoxon


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
            * "effect_size" : Cohen's d or Cohen's dz, a standardized measure of
              effect size indicating the magnitude of the difference between groups.
              For independent samples, Cohen's d is computed as the pooled or
              unpooled difference in means divided by the pooled or standard
              deviation, depending on ``assume_equal_variance``. Values are
              typically interpreted as: |d| < 0.2 (negligible), 0.2-0.5 (small),
              0.5-0.8 (medium), > 0.8 (large). For paired samples, Cohen's dz is
              used, which is the mean difference divided by the standard deviation
              of the differences.
            * "pvalue" : p-value of the test.
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    if paired:
        return _ttest_paired(arr1, arr2, alternative, ci, axis)
    return _ttest_ind(arr1, arr2, assume_equal_variance, alternative, ci, axis)


def rank_test(arr1, arr2, paired=False, alternative="two-sided", zero_method="wilcox", axis=0):
    """
    Perform a nonparametric rank-based comparison between two samples using either 
    the Mann-Whitney U test for independent samples or the Wilcoxon signed-rank 
    test for paired samples. The function returns the test statistic, p-value,
    and a rank-biserial correlation effect size ranging from -1 to 1, indicating
    the direction and magnitude of the difference.

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
    zero_method : {"wilcox", "pratt", "zsplit"}, optional
        Method used to handle zero differences in the Wilcoxon signed-rank test. 
        This parameter is only relevant when ``paired=True``.

        * "wilcox" - discard zero differences before ranking (default).
        * "pratt" - include zero differences when assigning ranks, but
          exclude their ranks from the positive and negative rank sums.
        * "zsplit" - include zero differences in the ranking and split
          their ranks equally between the positive and negative rank sums.
    axis : int, optional
        Axis corresponding to observations. If ``axis=0`` (default), compute one
        test per column. If ``axis=1``, compute one test per row.

    Returns
    -------
    dict
        dict containing the result of the selected statistical test:
        * "statistic" - The test statistic returned by the underlying Mann-Whitney U or
            Wilcoxon signed-rank test.
        * "pvalue" - The p-value associated with the test statistic and the specified
            ``alternative`` hypothesis.
        * "effect_size"`` - Rank-biserial correlation measuring the direction and magnitude
            of the difference between the two samples. The value ranges from -1 to 1.

            For the independent-samples Mann-Whitney U test, the
            rank-biserial correlation is calculated from the U statistic
            as:

                r_rb = 1 - (2 * U) / (n1 * n2)

            where ``n1`` and ``n2`` are the sample sizes. Positive values
            indicate that observations in ``arr1`` tend to be larger than
            those in ``arr2``; negative values indicate the reverse.

            For the paired Wilcoxon signed-rank test, the rank-biserial
            correlation is calculated as:

                r_rb = (W+ - W-) / (W+ + W-)

            where ``W+`` and ``W-`` are the sums of the positive and
            negative ranks, respectively. Positive values indicate that
            positive paired differences have greater rank sums, while
            negative values indicat
        
    Notes
    -----
    Tied absolute differences receive their average rank, matching SciPy's
    implementation of the Wilcoxon signed-rank test.

    The three zero-handling methods differ in when zero differences are
    included in the ranking:

    "wilcox"
        Zero differences are removed before ranking. Consequently, they have
        no effect on the ranks of the non-zero differences.

    "pratt"
        Zero differences remain when ranks are assigned. Their ranks are then
        excluded from both the positive and negative rank sums. Because the
        zeros participate in ranking, they can change the ranks assigned to
        the non-zero differences.

    "zsplit"
        Zero differences remain in the ranking and their ranks are split
        equally between the positive and negative rank sums.
    """
    arr1, arr2 = np.asarray(arr1), np.asarray(arr2)
    if paired:
        return _wilcoxon(arr1, arr2, alternative=alternative, axis=axis, zero_method=zero_method)
    return _mann_whitney(arr1, arr2, alternative=alternative, axis=axis)


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
