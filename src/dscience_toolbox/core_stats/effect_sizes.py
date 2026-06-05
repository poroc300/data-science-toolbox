"""Statistical inference functions for hypothesis testing."""

import numpy as np

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
