"""Private functions to apply nonparametric statistical tests."""

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
from scipy import stats


def _rank_biserial_correlation_independent(arr1, arr2):
    """
    Calculate rank-biserial correlation for independent samples (Mann-Whitney U).
    This is the effect size for the Mann-Whitney U test.

    Parameters
    ----------
    arr1 : ndarray
        Array containing observations from the first sample.
    arr2 : ndarray
        Array containing observations from the second sample.

    Returns
    -------
    float
        Rank-biserial correlation coefficient.
    """
    n1, n2 = arr1.size, arr2.size
    U = mannwhitneyu(arr1, arr2, alternative='two-sided').statistic
    r = 1 - (2 * U) / (n1 * n2)
    return r


def _rank_biserial_correlation_paired(arr1, arr2):
    """
    Calculate rank-biserial correlation for paired samples (Wilcoxon signed-rank).
    This is the effect size for the Wilcoxon signed-rank test.

    Parameters
    ----------
    arr1 : ndarray
        Array containing observations from the first paired measurement.
    arr2 : ndarray
        Array containing observations from the second paired measurement.

    Returns
    -------
    float
        Rank-biserial correlation coefficient.
    """
    diff = arr1 - arr2
    n = np.sum(diff != 0)  # number of non-zero differences
    if n == 0:
        return 0
    T = wilcoxon(arr1, arr2, alternative='two-sided').statistic
    r = 1 - (2 * T) / (n * (n + 1))
    return r


def _mannwhitneyu_ci(arr1, arr2, ci=0.95):
    """
    Calculate confidence interval for Mann-Whitney U test using the Hodges-Lehmann
    estimator (median of all pairwise differences).

    Parameters
    ----------
    arr1 : ndarray
        Array containing observations from the first sample.
    arr2 : ndarray
        Array containing observations from the second sample.
    ci : float
        Confidence level (default 0.95 for 95% CI).

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of the confidence interval.
    """
    n1, n2 = arr1.size, arr2.size
    
    # Calculate all pairwise differences
    differences = arr1[:, np.newaxis] - arr2[np.newaxis, :]
    differences = differences.flatten()
    differences = np.sort(differences)
    
    # Determine the rank indices for confidence interval
    alpha = 1 - ci
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    
    # Calculate critical index using normal approximation
    U_critical = stats.norm.ppf(ci) * np.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
    r1 = int(np.ceil((n1 * n2 / 2) - U_critical))
    r2 = int(np.floor((n1 * n2 / 2) + U_critical + 1))
    
    # Ensure indices are within bounds
    r1 = max(0, r1)
    r2 = min(len(differences) - 1, r2)
    
    return (differences[r1], differences[r2])


def _wilcoxon_ci(arr1, arr2, ci=0.95):
    """
    Calculate confidence interval for Wilcoxon signed-rank test using the Hodges-Lehmann
    estimator (median of pairwise differences of pairs).

    Parameters
    ----------
    arr1 : ndarray
        Array containing observations from the first paired measurement.
    arr2 : ndarray
        Array containing observations from the second paired measurement.
    ci : float
        Confidence level (default 0.95 for 95% CI).

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of the confidence interval.
    """
    diff = arr1 - arr2
    n = np.sum(diff != 0)
    
    if n == 0:
        return (0, 0)
    
    # Calculate all pairwise averages of differences
    pairwise_avg = []
    for i in range(n):
        for j in range(i, n):
            pairwise_avg.append((diff[i] + diff[j]) / 2)
    
    pairwise_avg = np.sort(np.array(pairwise_avg))
    
    # Calculate critical value using normal approximation
    alpha = 1 - ci
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    
    # Calculate critical index
    T_critical = z_alpha * np.sqrt((n * (n + 1) * (2 * n + 1)) / 24)
    r1 = int(np.ceil((n * (n + 1) / 4) - T_critical))
    r2 = int(np.floor((n * (n + 1) / 4) + T_critical + 1))
    
    # Ensure indices are within bounds
    r1 = max(0, r1)
    r2 = min(len(pairwise_avg) - 1, r2)
    
    return (pairwise_avg[r1], pairwise_avg[r2])


def _mannwhitneyu(arr1, arr2, alternative, ci, axis):
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
    ci : float
        Confidence level for the confidence interval. Must be between 0 and 1.
    axis : int
        Axis corresponding to observations. If ``axis=0``, compute one
        Mann-Whitney U test per column. If ``axis=1``, compute one test per row.

    Returns
    -------
    dict
        Dictionary containing:
            * "statistic" : Mann-Whitney U test statistic.
            * "estimate" : Difference in sample medians (Hodges-Lehmann estimator).
            * "ci" : Confidence interval for the median difference.
            * "effect_size" : Rank-biserial correlation (independent samples).
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

    # calculate effect size and confidence interval, return output as dict
    effect_size = _rank_biserial_correlation_independent(arr1, arr2)
    ci_bounds = _mannwhitneyu_ci(arr1, arr2, ci=ci)
    
    return {
        "statistic": result.statistic,
        "estimate": estimate,
        "ci": ci_bounds,
        "effect_size": effect_size,
        "pvalue": result.pvalue,
    }


def _wilcoxon(arr1, arr2, alternative, ci, axis):
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
    ci : float
        Confidence level for the confidence interval. Must be between 0 and 1.
    axis : int
        Axis corresponding to observations. If ``axis=0``, compute one
        Wilcoxon test per column. If ``axis=1``, compute one test per row.

    Returns
    -------
    dict
        Dictionary containing:
            * "statistic" : Wilcoxon signed-rank test statistic.
            * "estimate" : Difference in sample medians (Hodges-Lehmann estimator).
            * "ci" : Confidence interval for the median difference.
            * "effect_size" : Rank-biserial correlation (paired samples).
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

    # calculate effect size and confidence interval, return output as dict
    effect_size = _rank_biserial_correlation_paired(arr1, arr2)
    ci_bounds = _wilcoxon_ci(arr1, arr2, ci=ci)
    
    return {
        "statistic": result.statistic,
        "estimate": estimate,
        "ci": ci_bounds,
        "effect_size": effect_size,
        "pvalue": result.pvalue,
    }
