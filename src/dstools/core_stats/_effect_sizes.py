"""Private functions to compute effect sizes."""

import numpy as np


def _cohens_d_independent(arr1, arr2, axis=0):
    """
    Cohen's d for two independent groups (pooled standard deviation). Used by
    independent-samples t-test.

    Cohen's d is a standardized measure of the difference between two sample
    means. The effect size is obtained by dividing the difference in sample
    means by the pooled standard deviation, which combines the variability
    of both samples while accounting for their respective degrees of freedom.

    Parameters
    ----------
    arr1 : ndarray
        1D or 2D array containing observations from the first sample.
    arr2 : ndarray
        1D or 2D array containing observations from the second sample
    axis : int, optional
        Axis corresponding to observations. If ``axis=0`` (default), compute one Cohen's d
        per column. If ``axis=1``, compute one Cohen's d per row.

    Returns
    -------
    float
        Cohen's d effect size(s).
    """
    # unbiased sample variances
    n1 = arr1.shape[axis]
    n2 = arr2.shape[axis]
    var1 = np.var(arr1, axis=axis, ddof=1)
    var2 = np.var(arr2, axis=axis, ddof=1)

    # pooled variance weighted by each sample's degrees of freedom
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)

    # convert pooled variance back to standard deviation units
    pooled_sd = np.sqrt(pooled_var)

    # observed difference between sample means
    mean_diff = np.mean(arr1, axis=axis) - np.mean(arr2, axis=axis)

    # compute Cohen's d while safely handling zero pooled variance
    return np.divide(
        mean_diff,
        pooled_sd,
        out=np.zeros_like(mean_diff, dtype=float),
        where=pooled_sd != 0,
    )


def _cohens_dz_paired(arr1, arr2, axis=0):
    """
    Cohen's dz for paired (dependent) samples. Used by paired-samples t-test.

    Cohen's dz is a standardized measure of the mean difference between paired
    observations. The effect size is obtained by dividing the mean of the
    paired differences by the sample standard deviation of those differences.

    Parameters
    ----------
    arr1 : ndarray
        1D or 2D array containing observations from the first condition.
    arr2 : ndarray
        1D or 2D array containing observations from the second condition.
    axis : int, optional
        Axis corresponding to observations. If ``axis=0`` (default), compute one Cohen's
        dz per column. If ``axis=1``, compute one Cohen's dz per row.

    Returns
    -------
    float
        Cohen's dz effect size(s).
    """
    # paired differences
    diff = arr1 - arr2

    # sample standard deviation of the paired differences
    diff_sd = np.std(diff, axis=axis, ddof=1)

    # mean paired difference
    mean_diff = np.mean(diff, axis=axis)

    # compute Cohen's dz while safely handling zero standard deviation
    return np.divide(
        mean_diff,
        diff_sd,
        out=np.zeros_like(mean_diff, dtype=float),
        where=diff_sd != 0,
    )


def _rank_biserial_correlation_independent(mann_u, n1, n2):
    """
    Calculate rank-biserial correlation for independent samples (Mann-Whitney U).

    This is the effect size for the Mann-Whitney U test, ranging from -1 to 1.
    Positive values indicate that observations in the first sample tend to be
    larger than observations in the second sample; negative values indicate
    the reverse; 0 indicates no tendency either way.

    Parameters
    ----------
    mann_u : float or ndarray
        The Mann-Whitney U statistic, computed with respect to the first
        sample. If computed along an axis for multiple simultaneous tests,
        this is an array with that axis removed.
    n1 : int or ndarray
        Size of the first sample along the tested axis. Must be broadcastable
        against ``mann_u``.
    n2 : int or ndarray
        Size of the second sample along the tested axis. Must be broadcastable
        against ``mann_u``.

    Returns
    -------
    float or ndarray
        Rank-biserial correlation coefficient(s), matching the shape of ``mann_u``.
    """
    return 1 - (2 * mann_u) / (n1 * n2)
