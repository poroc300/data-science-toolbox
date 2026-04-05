"""Functions to assess correlation and multicollinearity among avriables."""

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def vif(df):
    """
    Compute Variance Inflation Factor (VIF) for each numeric variable. Low values
    indicate low multicollinearity (VIF < 5).

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing the variables to assess. Non-numeric columns
        are ignored automatically and rows with any NaN values are dropped before
        computation.

    Returns
    -------
    DataFrame
           Single-row DataFrame with the VIF for each numerical variable.

    """
    # get numeric columns and remove nan values
    df_num = df.select_dtypes("number").copy()
    df_num = df_num.dropna(axis=0, how="any")

    # estimate VIF for each variable (function only accepts numpy arrays)
    arr = df_num.to_numpy().copy()
    vif_ratios = []
    for idx in range(df_num.shape[1]):
        vif_ratios.append(variance_inflation_factor(arr, idx))

    # convert vif_ratios to DataFrame
    return pd.DataFrame([vif_ratios], columns=df_num.columns, index=["VIF"])
