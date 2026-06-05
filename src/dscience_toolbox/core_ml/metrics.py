"""Functions to compute metrics of machine-learning models."""

import numpy as np
from sklearn.metrics import r2_score

from dscience_toolbox.utils import _to_numpy


def percent_error(true, pred, absolute=False):
    """
    Calculate element-wise % error between two arrays.

    Parameters
    ----------
    true : int, float, array-like
        Ground truth values. Must not contain zeros (percent error is undefined
        when the true value is zero).
    pred : int, float, array-like
        Predicted or estimated values. Must be broadcast-compatible with true.
    absolute : bool, optional
        If True, returns the absolute % error (magnitude only). If False (default),
        returns signed % error, where positive values indicate over-prediction and
        negative values indicate under-prediction.

    Returns
    -------
    ndarray
        Element-wise % error values. Array's shape is determined by numpy broadcasting
        rules.
    """
    # check for zeros in true array
    if np.any(true == 0):
        raise ValueError("true contains zeros (percent error is undefined)")

    # calculate % error element-wise
    true, pred = _to_numpy(true), _to_numpy(pred)
    perc_error = ((pred - true) / np.abs(true)) * 100
    if absolute:
        return np.abs(perc_error)
    return perc_error


def r2_score_adj(true, pred, n_predictors):
    """
    Compute the adjusted coefficient of determination (adjusted R²).

    Parameters
    ----------
    true : array-like
        Ground truth target values.
    y_pred : array-like
        Predicted target values.
    n_predictors : int
        Number of predictors used in the model (excluding the intercept).

    Returns
    -------
    float or ndarray
        Adjusted R² score value or an array witl multiple adjusted R² values
        in case the input arrays have multiple columns.

    """
    # coerce to numpy arrays and get sample size
    true, pred = _to_numpy(true), _to_numpy(pred)
    if true.shape != pred.shape:
        raise ValueError("true and pred arrays must have the same shape")
    
    # raise errors if n_predictors is not an int and if sample size > n_predictors
    n = true.shape[0]
    if not isinstance(n_predictors, int):
        raise TypeError("n_predictors must be an int")
    if n <= n_predictors + 1:
        raise ValueError("sample size must be greater than n_predictors + 1")
    
    # calculate adjusted R2
    r2 = r2_score(true, pred, multioutput="raw_values")
    return 1 - (1 - r2) * (n - 1) / (n - n_predictors - 1)