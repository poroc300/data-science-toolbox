"""Functions to assess correlation and multicollinearity among variables."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr
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


def correlation(df, method="all"):
    """
    Compute correlation matrices and respective p-values among numerical variables.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing the variables to assess. Non-numeric columns
        are ignored automatically
    method : str, optional
        Correlation method to be used. Can be "all", "pearson", "spearman" or
        "kendall". The value "all" uses all three methods. The default is "all".

    Returns
    -------
    corr_results : dict
        dict containing DataFrames with correlation values and respective p-values
        among pairs of variables.

    """
    # find method to be used
    methods = ["pearson", "spearman", "kendall"]
    if method in methods:
        methods = [method]
    elif method != "all":
        raise ValueError("invalid method supplied")

    # functions to compute p-values per pair of variables
    pval_funcs = {
        "pearson": pearsonr,
        "spearman": spearmanr,
        "kendall": kendalltau,
    }

    # get copy of input DataFrame with only numeric columns
    df_num = df.select_dtypes("number").copy()

    # compute correlations and respective p-values
    cols = df_num.columns
    ncols = len(cols)
    corr_results = {}
    for method in methods:
        # pairwise correlation matrix
        corr_results[method] = df_num.corr(method=method)

        # build pairwise p-value matrix
        pval_matrix = np.ones((ncols, ncols))
        for i, j in combinations(range(ncols), 2):
            pair = df_num[[cols[i], cols[j]]].dropna()
            _, pval = pval_funcs[method](pair[cols[i]], pair[cols[j]])
            pval_matrix[i, j] = pval_matrix[j, i] = pval
        
        # add p-value matrix to output
        corr_results[f"{method}_pvalues"] = pd.DataFrame(
            pval_matrix, index=cols, columns=cols
        )

    return corr_results
