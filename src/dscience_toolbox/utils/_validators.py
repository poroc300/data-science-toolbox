"""Functions to enforce or validate inputs."""

import warnings

import numpy as np
import pandas as pd


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


def _validate_pandas_dataframe(df, input_arg):
    """
    Check if a pandas DataFrame was passed in a specific input argument. Will
    coerce a pandas Series to DataFrame.

    Parameters
    ----------
    df : DataFrame or Series
        Input object to validate.
    input_arg : str
        Name of input argument to be displayed on error message in case the
        input is not a Series or DataFrame.

    Returns
    -------
    pandas.DataFrame
        Validated DataFrame.
    """
    # input is a DataFrame
    if isinstance(df, pd.DataFrame):
        return df

    # input is a Series, coerce to a DataFrame
    if isinstance(df, pd.Series):
        warnings.warn("input Series was coerced to a DataFrame")
        return df.to_frame()

    # raise error when input is not a DataFrame or Series
    raise TypeError(
        f"{input_arg} must be a pandas DataFrame or Series, got {type(df).__name__}"
    )
