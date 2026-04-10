"""Functions to provide descriptive statistics."""

import numpy as np


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
        default is to use the 0.025, .25, .5, .75, .975 percentiles.

    Returns
    -------
    pd.DataFrame
        Stats table with columns = selected numeric columns.
    """
    # select only numeric columns
    df_num = df.select_dtypes(include="number").copy()
    if df_num.empty:
        raise ValueError("no numeric columns found")

    # check if provided subset of columns is valid
    if subset is not None:
        missing_cols = set(subset) - set(df_num.columns)
        if missing_cols:
            raise ValueError(f"Columns not found or non-numeric: {missing_cols}")
        df_num = df_num[subset]

    # default percentiles
    if percentiles is None:
        percentiles = (0.025, 0.25, 0.5, 0.75, 0.975)

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
