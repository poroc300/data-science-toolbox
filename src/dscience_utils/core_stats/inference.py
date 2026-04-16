"""Statistical inference functions for hypothesis testing."""

from scipy.stats import ttest_ind, ttest_rel


def t_test(arr1, arr2, paired=False, alternative="two-sided", **kwargs):
    """
    Perform a t-test (independent or paired) between two samples. This function
    wraps scipy.stats.ttest_ind and scipy.stats.ttest_rel.

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
    tuple
        tuple with the computed t-statistic and the respective p-value.
    """
    if paired:
        if len(arr1) != len(arr2):
            msg = "both samples must have the same length in a paired t-test"
            raise ValueError(msg)
        return ttest_rel(arr1, arr2, alternative=alternative, **kwargs)
    return ttest_ind(arr1, arr2, alternative=alternative, **kwargs)
