"""Private functions to apply non-parametric statistical tests."""

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
from scipy import stats


def _mannwhitneyu_ci(arr1, arr2, ci=0.95, axis=0):
    """
    Calculate a confidence interval for the Mann-Whitney U test using the
    Hodges-Lehmann estimator.

    The Hodges-Lehmann estimator is the median of all pairwise differences
    between observations from the two samples. The confidence interval is
    calculated by inverting a normal approximation to the Mann-Whitney U
    statistic.

    Parameters
    ----------
    arr1 : ndarray
        Array containing observations from the first sample.

    arr2 : ndarray
        Array containing observations from the second sample. Must have the
        same shape as ``arr1`` except along ``axis``.

    ci : float, default=0.95
        Confidence level for the interval. Must be between 0 and 1.

    axis : int, default=0
        Axis corresponding to observations. If ``axis=0``, compute one
        confidence interval per column. If ``axis=1``, compute one
        confidence interval per row.

    Returns
    -------
    tuple
        Tuple ``(lower_bound, upper_bound)`` containing the confidence
        interval bounds. Each bound has the same shape as the input arrays
        with ``axis`` removed.

    Notes
    -----
    Pairwise differences are calculated as ``arr1 - arr2``. Therefore,
    positive values indicate that observations from ``arr1`` tend to be
    larger than observations from ``arr2``.

    The confidence interval is based on a normal approximation to the
    Mann-Whitney U statistic.
    """
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1")

    axis = np.core.numeric.normalize_axis_index(axis, arr1.ndim)

    if arr1.shape[:axis] + arr1.shape[axis + 1 :] != (
        arr2.shape[:axis] + arr2.shape[axis + 1 :]
    ):
        raise ValueError("arr1 and arr2 must have matching shapes except along axis")

    # Move the observation axis to the front.
    arr1 = np.moveaxis(arr1, axis, 0)
    arr2 = np.moveaxis(arr2, axis, 0)

    n1 = arr1.shape[0]
    n2 = arr2.shape[0]

    if n1 == 0 or n2 == 0:
        raise ValueError("arr1 and arr2 must not be empty")

    output_shape = arr1.shape[1:]

    lower = np.empty(output_shape, dtype=float)
    upper = np.empty(output_shape, dtype=float)

    alpha = 1 - ci
    z = stats.norm.ppf(1 - alpha / 2)

    # Standard deviation of the Mann-Whitney U statistic under the null.
    u_sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)

    u_mean = n1 * n2 / 2

    lower_index = int(np.ceil(u_mean - z * u_sd))
    upper_index = int(np.floor(u_mean + z * u_sd))

    # Convert from U values to zero-based indices.
    lower_index = np.clip(lower_index, 0, n1 * n2 - 1)
    upper_index = np.clip(upper_index, 0, n1 * n2 - 1)

    for index in np.ndindex(output_shape):
        sample1 = arr1[(slice(None),) + index]
        sample2 = arr2[(slice(None),) + index]

        differences = np.sort((sample1[:, np.newaxis] - sample2[np.newaxis, :]).ravel())

        lower[index] = differences[lower_index]
        upper[index] = differences[upper_index]

    return lower, upper


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
