"""Functions to compute metrics of machine-learning models."""

import numpy as np

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
