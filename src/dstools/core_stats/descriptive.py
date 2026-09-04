"""Functions to provide descriptive statistics."""

import numpy as np
import pandas as pd

from dstools.utils import _validate_pandas_dataframe


def column_dtypes(df):
    """
    Get a dict grouping the columns per dtype.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.

    Returns
    -------
    dict
        dict mapping the dtype names to the column names (lists) with that dtype.
    """
    df = _validate_pandas_dataframe(df, input_arg="df")
    _dtypes = df.dtypes.astype(str)
    return _dtypes.groupby(_dtypes).apply(lambda x: x.index.tolist()).to_dict()


def summary_stats(df, subset=None, percentiles=None):
    """
    Descriptive statistics built on top of pd.DataFrame.describe(), extended with
    skewness, kurtosis, coefficient of variation, and missing counts.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    subset : list, optional
        Subset of columns to summarise. Defaults to all numeric columns.
    percentiles : list, optional
        Quantiles passed directly to describe() (values between 0 and 1). The
        default is to use the 0.025, .10, .25, .5, .75, .90, .975 percentiles.

    Returns
    -------
    pd.DataFrame
        Stats table with columns = selected numeric columns.
    """
    # select only numeric columns and remove those with only NaN
    df = _validate_pandas_dataframe(df, input_arg="df")
    df_num = df.select_dtypes(include="number").copy()
    df_num = df_num.dropna(axis=1, how="all")
    if df_num.empty:
        raise ValueError("no numeric columns found")

    # check if provided subset of columns is valid
    if subset is not None:
        missing_cols = set(subset) - set(df_num.columns)
        if missing_cols:
            raise ValueError(f"columns not found or non-numeric: {missing_cols}")
        df_num = df_num[subset]

    # default percentiles
    if percentiles is None:
        percentiles = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)

    # descriptive statistics estimated by pandas decribe()
    summary_df = df_num.describe(percentiles=percentiles)
    summary_df.index.name = "stat"

    # get number and frequency of NaN, skewness, kurtosis and coefficient of variation
    summary_df.loc["missing"] = df_num.isna().sum(axis=0)
    summary_df.loc["missing_%"] = df_num.isna().mean() * 100
    summary_df.loc["skewness"] = df_num.skew(axis=0, skipna=True)
    summary_df.loc["kurtosis"] = df_num.kurt(axis=0, skipna=True)
    summary_df.loc["CV_%"] = (df_num.std() / df_num.mean().replace(0, np.nan)) * 100
    return summary_df.round(4)


def categorical_summary_stats(df, subset=None):
    """
    Descriptive statistics for categorical columns. Outputs count, missing value
    statistics, number of unique categories, and the most frequent category.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    subset : list, optional
        Subset of categorical columns to summarise.

    Returns
    -------
    summary : DataFrame
        Summary statistics for categorical variables.
    """
    # select only categorical columns
    df = _validate_pandas_dataframe(df, input_arg="df")
    cat_types = ("object", "category", "string", "bool")
    df_cat = df.select_dtypes(include=cat_types).copy()
    df_cat = df_cat.dropna(axis=1, how="all")
    if df_cat.empty:
        raise ValueError("no categorical columns found")

    # check if provided subset of columns is valid
    if subset is not None:
        missing_cols = set(subset) - set(df_cat.columns)
        if missing_cols:
            raise ValueError(f"columns not found or non-categorical: {missing_cols}")
        df_cat = df_cat[subset]

    # empty DataFrame to store results
    summary = pd.DataFrame(columns=df_cat.columns)

    # basic counts
    summary.loc["count"] = df_cat.notna().sum()
    summary.loc["missing"] = df_cat.isna().sum()
    summary.loc["missing_%"] = df_cat.isna().mean() * 100

    # unique number of categories
    summary.loc["n_unique"] = df_cat.nunique(dropna=True)

    # top category and frequency
    summary.loc["top"] = df_cat.apply(lambda x: x.mode().iloc[0])
    summary.loc["top_freq"] = df_cat.apply(
        lambda x: x.value_counts(normalize=True, dropna=True).iloc[0] * 100
    )
    return summary
