"""Miscellaneous mathematic functions."""

import numpy as np

from dscience_toolbox.utils import _to_numpy


def rceil(arr, dec=None):
    """
    Apply ceil rounding to scalars or arrays to a specific decimal precision.

    Parameters
    ----------
    arr : int, float, array-like
        Input values.
    dec : int or None, optional
        Number of decimal places to round to. If None or 0, values are
        rounded up to the nearest integer using numpy.ceil.

    Returns
    -------
    ndarray or scalar
        Array or scalar with rounded values.

    """
    # coerce to numpy object
    arr = _to_numpy(arr)

    # invalid dec values
    if dec is not None:
        if not isinstance(dec, int) or dec < 0:
            raise TypeError("dec must be None or a positive int")

    # default to np.ceil
    if dec is None or dec == 0:
        return np.ceil(arr)

    # decimal ceil rounding
    factor = 10**dec
    return np.ceil(arr * factor) / factor


def rfloor(arr, dec=None):
    """
    Apply floor rounding to scalars or arrays to a specific decimal precision.

    Parameters
    ----------
    arr : int, float, array-like
        Input values.
    dec : int or None, optional
        Number of decimal places to round to. If None or 0, values are
        rounded down to the nearest integer using numpy.floor.

    Returns
    -------
    ndarray or scalar
        Array or scalar with rounded values.

    """
    # coerce to numpy object
    arr = _to_numpy(arr)

    # invalid dec values
    if dec is not None:
        if not isinstance(dec, int) or dec < 0:
            raise TypeError("dec must be None or a positive int")

    # default to np.ceil
    if dec is None or dec == 0:
        return np.floor(arr)

    # decimal ceil rounding
    factor = 10**dec
    return np.floor(arr * factor) / factor
