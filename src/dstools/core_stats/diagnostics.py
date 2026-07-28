"""Functions for common data diagnostics and assumption testing."""

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import (
    anderson,
    kendalltau,
    kurtosis,
    normaltest,
    pearsonr,
    shapiro,
    skew,
    spearmanr,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

from dstools.utils import _validate_pandas_dataframe


def vif(df):
    """
    Compute Variance Inflation Factor (VIF) for each numeric variable. Low values
    indicate low multicollinearity (VIF < 5).

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing the variables to assess. Non-numeric columns
        are ignored automatically and rows with any NaN values and constante values
        are dropped before computation.

    Returns
    -------
    DataFrame
        Single-row DataFrame with the VIF for each numerical variable.

    Notes
    -----
    Pre-processing

    """
    # get numeric columns, remove columns with constant and nan values
    df = _validate_pandas_dataframe(df, input_arg="df")
    df_num = df.select_dtypes("number").copy()
    df_num = df_num.dropna(axis=0, how="any")
    df_num = df_num.loc[:, df_num.nunique() > 1]
    if df_num.empty:
        raise ValueError("preprocessing made the input DataFrame empty")

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
    df = _validate_pandas_dataframe(df, input_arg="df")
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


def check_normality(arr):
    """
    Assess the normality of a distribution using three complementary normality
    tests (Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling) alongside skewness
    and kurtosis to provide both hypothesis-test results and descriptive diagnostics.

    Parameters
    ----------
    arr : array-like
        Input data. Must be a 1D sequence.

    Returns
    -------
    dict
        For the three tests, the dict will contain the keys "stat" and "pval"
        which are the statistic value and p-value, respectively. For kurtosis and
        skeweness only their respective value is shown.

    Notes
    -----
    These are more details about the tests and statistics used in this function:
        - Shapiro-Wilk: Most powerful for small samples (n < 50).
        - D'Agostino-Pearson: Combines skewness and kurtosis into a single
        chi-squared statistic.
        - Anderson-Darling: Sensitive to tail deviations. Suitable for medium
        to large samples.
        - kurtosis: a value of zero indicates similar tail behaviour to a normal
        distribution. Positive values indicate heavier tails, negative values
        indicate lighter tails.
        - skewness:  measure of asymmetry. 0 indicates a symmetric distribution.
        Positive values indicate a right-skewed distribution, negative values
        indicate left-skewed.

    All tests become hypersensitive when sample sizes are very large (e.g. > 2000).
    In those casesm consider using skewness and kurtosis values combined with visual
    inspection (e.g. Q-Q plot) instead. A warning is printed when the sample size
    is > 2000.

    """
    # coerce to numpy array
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("arr must be a 1D array-like")
    if arr.size > 2000:
        warnings.warn("the sample size is high, tests may be oversensitive")

    # dict to store results of tests
    results = {}

    # Shapiro-Wilk (good for small samples n < 50)
    stat, p = shapiro(arr)
    results["shapiro_wilk"] = {"stat": stat, "pval": p}

    # D'Agostino-Pearson omnibus test
    stat, p = normaltest(arr)
    results["dagostino_pearson"] = {"stat": stat, "pval": p}

    # Anderson-Darling test
    stat, p = anderson(arr, dist="norm", method="interpolate")
    results["anderson_darling"] = {"stat": stat, "pval": p}

    # skewness and kurtosis
    results["skewness"] = skew(arr)
    results["kurtosis"] = kurtosis(arr)
    return results
