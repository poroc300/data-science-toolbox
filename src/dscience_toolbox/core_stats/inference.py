"""Statistical inference functions for hypothesis testing."""

import warnings

import numpy as np
from scipy.stats import (
    anderson,
    kurtosis,
    mannwhitneyu,
    normaltest,
    permutation_test,
    shapiro,
    skew,
    ttest_ind,
    ttest_rel,
    wilcoxon,
)

from dscience_toolbox.utils import _to_numpy


def check_normality(arr):
    """
    Assess the normality of a distribution using three complementary normality
    tests (Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling) alongside skewness
    and kurtosis to provide both hypothesis-test results and descriptive diagnostics.

    Parameters
    ----------
    arr : array-like
        Input data. Will be coerced to a 1D numpy array. Accepts also lists and
        pandas Series.

    Returns
    -------
    dict
        For the three tests, the dict will contain the keys "stat" and "pval"
        which are the statistic value and p-value, respectively. For kurtosis and
        skeweness only their respective value is shown.

    Notes
    -----
    These are more details about the tests and statistics used in this function:
        - Shapiro-Wilk: Most powerful for small samples (n < 50).
        - D'Agostino-Pearson: Combines skewness and kurtosis into a single
        chi-squared statistic.
        - Anderson-Darling: Sensitive to tail deviations. Suitable for medium
        to large samples.
        - kurtosis: a value of zero indicates similar tail behaviour to a normal
        distribution. Positive values indicate heavier tails, negative values
        indicate lighter tails.
        - skewness:  measure of asymmetry. 0 indicates a symmetric distribution.
        Positive values indicate a right-skewed distribution, negative values
        indicate left-skewed.

    All tests become hypersensitive when sample sizes are very large (e.g. > 2000).
    In those casesm consider using skewness and kurtosis values combined with visual
    inspection (e.g. Q-Q plot) instead. A warning is printed when the sample size
    is > 2000.

    """
    # coerce to numpy array and flat array to 1D
    arr = _to_numpy(arr, ndim=1)
    if arr.size > 2000:
        warnings.warn("the sample size is high, tests may be oversensitive")

    # dict to store results of tests
    results = {}

    # Shapiro-Wilk (good for small samples n < 50)
    stat, p = shapiro(arr)
    results["shapiro_wilk"] = {"stat": stat, "pval": p}

    # D'Agostino-Pearson omnibus test
    stat, p = normaltest(arr)
    results["dagostino_pearson"] = {"stat": stat, "pval": p}

    # Anderson-Darling test
    stat, p = anderson(arr, dist="norm", method="interpolate")
    results["anderson_darling"] = {"stat": stat, "pval": p}

    # skewness and kurtosis
    results["skewness"] = skew(arr)
    results["kurtosis"] = kurtosis(arr)
    return results


def _cohens_d_independent(arr1, arr2):
    """
    Cohen's d for two independent groups (pooled standard deviation). Used by
    independent-samples t-test.

    Parameters
    ----------
    arr1 : ndarray
        1D numpy array.
    arr2 : ndarray
        1D numpy array.

    Returns
    -------
    float
        Cohen's d effect size.
    """
    # sample sizes of input arrays
    n1, n2 = arr1.size, arr2.size

    # weight each variance by its degrees of freedom
    weighted_var1 = (n1 - 1) * np.var(arr1, ddof=1)
    weighted_var2 = (n2 - 1) * np.var(arr2, ddof=1)

    # sum the weighted variances and divide by total degrees of freedom
    pooled_var = (weighted_var1 + weighted_var2) / (n1 + n2 - 2)

    # square root to get back to SD units
    pooled_sd = np.sqrt(pooled_var)

    # effect size
    return (np.mean(arr1) - np.mean(arr2)) / pooled_sd


def _cohens_dz_paired(arr1, arr2):
    """
    Cohen's dz for paired/dependent samples using difference scores (dz). To be
    used by paired-samples t-test.

    Parameters
    ----------
    arr1 : ndarray
        1D numpy array (pre-test scores).
    arr2 : ndarray
        1D numpy array (post-test scores).

    Returns
    -------
    float
        Cohen's dz effect size for paired samples.
    """
    assert arr1.size == arr2.size, "unequal size of input arrays"
    diff = arr1 - arr2
    return np.mean(diff) / np.std(diff, ddof=1)


def cohens_d(arr1, arr2, method="independent"):
    """
    Calculate Cohen's effect size. Currently support Cohen's d for two independent
    groups (independent t-test) and Cohen's dz for two dependent samples (paired
    t-test).

    Parameters
    ----------
    arr1 : array-like
        1D array of the first group sample.
    arr2 : array-like
        1D array of the second group sample.
    method : str, optional
        Supports "independent" for independent Cohen's d and "paired" for Cohen's
        dz. The default is "independent".

    Returns
    -------
    float
        Cohen's effect size value.

    """
    # Cohen's d type
    methods = {"independent": _cohens_d_independent, "paired": _cohens_dz_paired}
    if method not in methods.keys():
        raise KeyError("invalid method")

    # calculate Cohen's d
    arr1, arr2 = _to_numpy(arr1, ndim=1), _to_numpy(arr2, ndim=1)
    cohens_d_func = methods[method]
    return cohens_d_func(arr1, arr2)


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
