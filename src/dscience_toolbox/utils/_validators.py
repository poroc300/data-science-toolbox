"""Functions to enforce or validate inputs."""

import warnings

import numpy as np


def _to_numpy(x, dtype="float64", ndim=None):
    """
    Convert input to numpy array and optionally enforce dimensions.

    Parameters
    ----------
    x : array-like or ndarray
        Array to be coerced. Handles int, floats, lists, pandas objects and
        numpy arrays.
    dtype : numpy dtype, optional
        numpy compatible dtype. Defaults to np.float64.
    ndim : int, optiona;
        Required dimensions, either 1 or 2 (not compatible with more higher number
        of dimensions). The default is to not change the dimension.

    Returns
    -------
    ndarray
        numpy array.
        
    Notes
    -----
    If a 2D array is supplied and required to coerce to 1D array, a warning will
    be printed in cases where multiple columns exist.        

    """
    # check input is a numpy array otherwise coerce it
    if isinstance(x, np.ndarray):
        arr = x.astype(dtype, copy=False)
    else:
        arr = np.asarray(x, dtype=dtype)

    # check array is empty
    if arr.size == 0 or arr.ndim > 2:
        raise ValueError("input cannot be empty or have > 2 dimensions")

    # flat 2D array to 1D array
    if ndim == 1:
        if arr.ndim == 2 and arr.shape[1] > 1:
            warnings.warn("multi-column array was flatten")
        arr = arr.ravel()
    # change 1D array to a 2D array
    elif ndim == 2 and arr.ndim <= 1:
        arr = arr.reshape(-1, 1)
    return arr
