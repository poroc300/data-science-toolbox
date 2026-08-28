"""Private functions to apply non-parametric statistical tests."""

import numpy as np
from scipy.stats import mannwhitneyu, rankdata, wilcoxon

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


def _wilcoxon_rank_sums(diff, zero_method, axis=0):
    """
    Calculate positive and negative rank sums for a Wilcoxon signed-rank test.

    The absolute values of the paired differences are ranked, and the ranks
    are then summed separately according to the sign of the corresponding
    differences. The resulting positive and negative rank sums can be used
    to calculate the rank-biserial correlation effect size.

    Parameters
    ----------
    diff : ndarray
        Array containing paired differences between two measurements. The
        dimension specified by ``axis`` corresponds to paired observations;
        any remaining dimensions represent independent tests.
    zero_method : {"wilcox", "pratt", "zsplit"}, default="wilcox"
        Method used to handle zero differences. See Notes for more details.
    axis : int, default=0
        Axis along which paired observations are located.

    Returns
    -------
    positive_rank_sum : float or ndarray
        Sum of ranks corresponding to positive differences.
    negative_rank_sum : float or ndarray
        Sum of ranks corresponding to negative differences.

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
    # check zero-handling method is valid
    if zero_method not in {"wilcox", "pratt", "zsplit"}:
        raise ValueError("zero_method must be one of 'wilcox', 'pratt', or 'zsplit'")

    # move the observation axis to the first position for consistent ranking
    # across observations, with remaining axes representing independent tests
    diff = np.moveaxis(diff, axis, 0)

    # for "wilcox", zeros are discarded BEFORE ranking; nan is used as a
    # temporary marker so that the observation positions are preserved while
    # allowing us to exclude them from rankdata
    if zero_method == "wilcox":
        diff = diff.astype(float, copy=True)
        diff[diff == 0] = np.nan

    # ranks are calculated based on the absolute differences
    abs_diff = np.abs(diff)

    # array with nan values to store
    ranks = np.full(abs_diff.shape, np.nan, dtype=float)

    # rankdata operates on one-dimensional arrays, so iterate over every
    # combination of the non-observation dimensions
    # for example, if diff has shape (50, 3), this loop ranks each of the
    # three columns independently
    for index in np.ndindex(abs_diff.shape[1:]):
        values = abs_diff[(slice(None),) + index]

        # remove nan values from ranking, tied ranks use average
        if zero_method == "wilcox":
            valid = ~np.isnan(values)
            ranks[(slice(None),) + index][valid] = rankdata(
                values[valid],
                method="average",
            )

        # for "pratt" and "zsplit", zero differences participate in the
        # ranking; rankdata assigns tied values their average rank
        else:
            ranks[(slice(None),) + index] = rankdata(
                values,
                method="average",
            )

    # select the ranks associated with positive differences and sum them
    # np.where ensures that ranks belonging to negative differences and zeros
    # contribute zero to W+
    positive_rank_sum = np.nansum(np.where(diff > 0, ranks, 0), axis=0)

    # calculate W-, the sum of ranks associated with negative differences
    negative_rank_sum = np.nansum(np.where(diff < 0, ranks, 0), axis=0)

    # zero differences are not discarded for "zsplit", ranks are split equally between W+ and W-
    if zero_method == "zsplit":
        zero_rank_sum = np.sum(np.where(diff == 0, ranks, 0), axis=0)
        positive_rank_sum += zero_rank_sum / 2
        negative_rank_sum += zero_rank_sum / 2

    return positive_rank_sum, negative_rank_sum


def _wilcoxon(arr1, arr2, alternative, zero_method, axis):
    """
    Perform a Wilcoxon signed-rank test to compare two paired samples.

    The test evaluates whether the paired differences are systematically
    positive or negative by ranking the absolute differences and accounting
    for their direction.

    Parameters
    ----------
    arr1 : ndarray
        Array containing observations from the first paired measurement.
    arr2 : ndarray
        Array containing observations from the second paired measurement.
    alternative : {"two-sided", "less", "greater"}
        Defines the alternative hypothesis.

        * ``"two-sided"``: the distributions of the paired differences differ.
        * ``"less"``: the first measurement tends to have smaller values than
          the second measurement.
        * ``"greater"``: the first measurement tends to have larger values than
          the second measurement.
    zero_method : {"wilcox", "pratt", "zsplit"}
        Method used to handle zero differences.

        * ``"wilcox"``: discard zero differences.
        * ``"pratt"``: include zero differences when ranking, but exclude their
          ranks from the test statistic.
        * ``"zsplit"``: include zero differences in the ranking and split their
          ranks equally between positive and negative differences.
    axis : int
        Axis corresponding to observations. If ``axis=0``, compute one test
        per column. If ``axis=1``, compute one test per row.

    Returns
    -------
    dict
        Dictionary containing:

        * "statistic" : Wilcoxon signed-rank test statistic.
        * "pvalue" : p-value corresponding to the Wilcoxon signed-rank test.
        * "effect_size" : Rank-biserial correlation measuring the direction and magnitude of
          the paired differences.

    Notes
    -----
    The test is performed using SciPy's ``wilcoxon`` implementation, including
    its handling of zero differences, ties, and other rank-related edge cases.
    """
    # samples must have same length
    if arr1.shape[axis] != arr2.shape[axis]:
        raise ValueError(
            "both samples must have the same number of observations "
            "along the specified axis in a paired test"
        )

    # perform Wilcoxon test
    result = wilcoxon(
        arr1,
        arr2,
        alternative=alternative,
        zero_method=zero_method,
        axis=axis,
    )

    # alculate the rank-biserial correlation from the paired differences.
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
