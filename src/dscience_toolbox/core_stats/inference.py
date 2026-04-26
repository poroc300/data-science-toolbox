"""Statistical inference functions for hypothesis testing."""

import numpy as np
from scipy.stats import ttest_ind, ttest_rel

from dscience_toolbox.utils import _to_numpy


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
    return (np.mean(arr1) - np.mean(arr1)) / pooled_sd


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
    diff = arr1 - arr2
    return np.mean(diff) / np.std(diff, ddof=1)


def cohens_d(arr1, arr2, method="independent"):
    """
    SUMMARY.

    Parameters
    ----------
    arr1 : TYPE
        DESCRIPTION.
    arr2 : TYPE
        DESCRIPTION.
    method : TYPE, optional
        DESCRIPTION. The default is "independent".

    Returns
    -------
    TYPE
        DESCRIPTION.

    Raises
    ------
    KeyError
        DESCRIPTION.
    """
    # Cohen's d type
    methods = {"independent": _cohens_d_independent, "paired": _cohens_dz_paired}
    if method not in methods.keys():
        raise KeyError("invalid method")
    
    # calculate Cohen's d
    arr1, arr2 = _to_numpy(arr1, ndim=1), _to_numpy(arr2, ndim=1)
    return round(methods[method](arr1, arr2), 1)

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
    arr1, arr2 = _to_numpy(arr1), _to_numpy(arr2)
    if paired:
        if len(arr1) != len(arr2):
            msg = "both samples must have the same length in a paired t-test"
            raise ValueError(msg)
        return ttest_rel(arr1, arr2, alternative=alternative, **kwargs)
    return ttest_ind(arr1, arr2, alternative=alternative, **kwargs)
