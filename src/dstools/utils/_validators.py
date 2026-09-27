"""Functions to enforce or validate inputs."""

import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


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


def _validate_numeric_array(data):
    """
    Check if provided input contains numeric data only. Output will depend of
    the supplied input types.

    Parameters
    ----------
    data : list, ndarray, Series, or DataFrame
        Input data to check for numeric data.

    Returns
    -------
    ndarray, Series or DataFrame
        Numeric array, preserving 1D/2D shape (minus any dropped non-numeric
        DataFrame columns). Returned type depends on input object types:
            * If input is ``pd.DataFrame``, output preserves object type with discarded
            non-numeric columns.
            * If input is ``pd.Series``, function preserves object type.
            * Function vasts to ``np.ndarray`` any other valid input types.
    """
    # DataFrames can have columns of different dtypes
    if isinstance(data, pd.DataFrame):
        numeric_data = data.select_dtypes(include=np.number).copy()
        dropped = set(data.columns) - set(numeric_data.columns)
        if dropped and numeric_data.shape[1] > 0:
            warnings.warn(
                f"dropping non-numeric column(s) {sorted(dropped)}",
                UserWarning,
                stacklevel=2,
            )
        elif numeric_data.shape[1] == 0:
            raise ValueError("input DataFrame has no numeric columns")

        return numeric_data

    # handle other object types
    elif isinstance(data, (list, tuple, np.ndarray, pd.Series)):
        if not isinstance(data, (pd.Series, np.ndarray)):
            data = np.asarray(data)
        if len(data) == 0:
            raise ValueError("an empty sequence was supplied")
        elif not is_numeric_dtype(data):
            raise ValueError("input array/sequence does not have a numeric dtype")

        return data

    # unspported object input types
    raise TypeError(
        f"unsupported type {type(data).__name__!r}; expected list, tuple, "
        "ndarray, Series, or DataFrame"
    )


def _check_plotting_inputs(df, required_cols):
    """
    Check the input variables provided to a plotting function.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame provided to the plotting function.
    required_cols : sequence-like
        Sequence with the names of the columns to be used (e.g. for x and y axes).

    Raises
    ------
    KeyError
        When a column name is not in df, a KeyError is raised.
    """
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise KeyError(f"missing columns in DataFrame: {missing_cols}")


def _merge_plot_kwargs(params, plot_kwargs):
    """
    Merge user-supplied plotting keyword arguments with internally defined
    plotting parameters. This function is intended to prevent users from
    inadvertently overriding parameters that are explicitly controlled by the
    plotting function while still allowing additional keyword arguments to be
    passed through to the underlying seaborn library.

    Parameters
    ----------
    params : dict
        Dictionary containing plotting parameters defined internally by the
        function. These values take precedence over any duplicated keys in
        ``plot_kwargs``.
    plot_kwargs : dict or None
        Additional keyword arguments to pass to the plotting function. If
        ``None``, ``params`` is returned unchanged.

    Returns
    -------
    dict
        Dictionary containing the merged plotting parameters. A warning is raised
        if duplicated keys exist between both dict inputs.

    """
    if plot_kwargs is None:
        return params
    if not isinstance(plot_kwargs, dict):
        raise TypeError("plot_kwargs must be a dict")

    # check for duplicated keys and raise warning if found
    duplicated = set(params) & set(plot_kwargs)
    if duplicated:
        warnings.warn(
            "ignoring values in plot_kwargs for duplicated parameter(s): "
            f"{', '.join(sorted(duplicated))}",
            UserWarning,
            stacklevel=2,
        )

    # merge keys (params override plot_kwargs()
    merged = plot_kwargs.copy()
    merged.update(params)
    return merged
