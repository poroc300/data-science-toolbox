"""Private functions to apply parametric statistical tests."""

import numpy as np
from scipy.stats import ttest_ind, ttest_rel

from dstools.core_stats._effect_sizes import _cohens_d_independent, _cohens_dz_paired


def _ttest_ind(arr1, arr2, assume_equal_variance, alternative, ci, axis):
    """
    Perform an independent t-test. This function compares the means of two samples
    under the assumption that the data are approximately normally distributed.

    Parameters
    ----------
    arr1 : ndarray
        Array containing observations from the first sample.
    arr2 : ndarray
        Array containing observations from the second sample.
    assume_equal_variances : bool
        Whether to assume that the two populations have equal variances when
        performing an independent-samples t-test. If False, perform
        Welch's t-test, which does not assume equal population variances and
        adjusts the degrees of freedom accordingly. If True, perform Student's
        t-test, which pools the sample variances to estimate a common population
        variance.
    alternative : {"two-sided", "less", "greater"}
        Defines the alternative hypothesis.
            * "two-sided": the population means are unequal.
            * "less": the mean of sample 1 is less than the mean of sample 2.
            * "greater": the mean of sample 1 is greater than the mean of sample 2.
    ci : float
        Confidence level for the confidence interval of the estimated mean
        difference. Must be between 0 and 1.
    axis : int
        Axis corresponding to observations. If ``axis=0``, compute one
        t-test per column. If ``axis=1``, compute one t-test per row.

    Returns
    -------
    dict
        Dictionary containing:
            * "statistic" : t-test statistic.
            * "estimate" : Difference in sample means.
            * "ci" : Confidence interval for the mean difference.
            * "effect_size" : Cohen's d (independent samples)
            * "pvalue" : p-value of the test.
    """
    # observed difference between sample means
    estimate = np.mean(arr1, axis=axis) - np.mean(arr2, axis=axis)

    # perform independent-samples t-test (Student's or Welch's)
    result = ttest_ind(
        arr1,
        arr2,
        equal_var=assume_equal_variance,
        alternative=alternative,
        axis=axis,
    )

    # calculate effect size and confidence interval, return output as dict
    effect_size = _cohens_d_independent(arr1, arr2, axis=axis)
    conf_int = result.confidence_interval(confidence_level=ci)
    return {
        "statistic": result.statistic,
        "estimate": estimate,
        "ci": (conf_int.low, conf_int.high),
        "effect_size": effect_size,
        "pvalue": result.pvalue,
    }


def _ttest_paired(arr1, arr2, alternative, ci, axis):
    """
    Perform a paired-samples t-test. This function compares the means of two samples
    under the assumption that the data are approximately normally distributed.

    Parameters
    ----------
    arr1 : ndarray
        Array containing observations from the first paired measurement.
    arr2 : ndarray
        Array containing observations from the second paired measurement.
    alternative : {"two-sided", "less", "greater"}
        Defines the alternative hypothesis.
            * "two-sided": the population means are unequal.
            * "less": the mean of sample 1 is less than the mean of sample 2.
            * "greater": the mean of sample 1 is greater than the mean of sample 2.
    ci : float
        Confidence level for the confidence interval of the estimated mean
        difference. Must be between 0 and 1.
    axis : int
        Axis corresponding to observations. If ``axis=0``, compute one
        t-test per column. If ``axis=1``, compute one t-test per row.

    Returns
    -------
    dict
        Dictionary containing:
            * "statistic" : t-test statistic.
            * "estimate" : Difference in sample means.
            * "confidence_interval" : Confidence interval for the mean difference.
            * "effect_size" : Cohen's dz (paired samples).
            * "pvalue" : p-value of the test.
    """
    # observed difference between sample means
    estimate = np.mean(arr1, axis=axis) - np.mean(arr2, axis=axis)

    # samples must have same length
    if arr1.shape[axis] != arr2.shape[axis]:
        raise ValueError(
            "both samples must have the same number of observations "
            "along the specified axis in a paired t-test"
        )

    # perform paired-samples t-test
    result = ttest_rel(
        arr1,
        arr2,
        alternative=alternative,
        axis=axis,
    )

    # calculate Cohen's dz and confidence interval, return output as dict
    effect_size = _cohens_dz_paired(arr1, arr2, axis=axis)
    conf_int = result.confidence_interval(confidence_level=ci)
    return {
        "statistic": result.statistic,
        "estimate": estimate,
        "ci": (conf_int.low, conf_int.high),
        "effect_size": effect_size,
        "pvalue": result.pvalue,
    }
