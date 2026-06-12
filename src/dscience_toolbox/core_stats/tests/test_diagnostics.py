"""Unit tests for functions in diagnostics.py."""

from contextlib import suppress

import numpy as np
import pandas as pd
import pytest

from dscience_toolbox.core_stats import check_normality, correlation, vif

# set random seed number
np.random.seed(7)


def test_vif(sample_df):
    """Test if low/high vif values are calculated correctly."""
    # test categorical columns are ignored and high/low multicollinearity
    test_vif = vif(sample_df)
    assert "x4" not in test_vif.columns
    assert np.all(test_vif[["x1", "x2"]] > 10)
    assert test_vif["x3"].iat[0] < 5


def test_vif_nan(sample_df):
    """Test if rows with nan values are discarded"""
    test_vif = vif(sample_df)
    sample_df.loc[0, "x1"] = np.nan
    test_vif_nan = vif(sample_df)
    assert np.all(test_vif_nan != test_vif)


def test_vif_empty_after_preprocessing():
    """Test that error is raised when input is empty after preprocessing."""
    df = pd.DataFrame({"x1": [1] * 2, "x2": [np.nan] * 2})
    with pytest.raises(ValueError, match="preprocessing made the input"):
        vif(df)


def test_correlation(sample_df):
    """Test if correlation values are calculated correctly."""
    # test high/low correlation levels
    results = correlation(sample_df, method="spearman")
    assert results["spearman"].loc["x1", "x2"] >= 0.99
    assert results["spearman"].loc["x1", "x3"] < 0.1
    assert results["spearman_pvalues"].loc["x1", "x2"] < 0.05
    assert results["spearman_pvalues"].loc["x1", "x3"] > 0.5

    # test all keys are returned when method="all"
    assert len(correlation(sample_df, method="all").keys()) == 6


def test_correlation_pvalue_diagonal(sample_df):
    """Test diagonal of p-value matrix is just 1."""
    results = correlation(sample_df, method="pearson")
    assert np.all(np.diag(results["pearson_pvalues"]) == 1)


def test_correlation_ignores_non_numeric(sample_df):
    """Test correlation ignores non-numeric columns."""
    results = correlation(sample_df, method="pearson")
    assert "x4" not in results["pearson"].columns


def test_correlation_invalid_method(sample_df):
    """Test error is raised when invalid method name is supplied."""
    with pytest.raises(ValueError, match="invalid method supplied"):
        correlation(sample_df, method="invalid")


def test_check_normality_normal_dist():
    """Test when input array exhibits normal distribution."""
    # create test array and get result
    arr = np.random.normal(50, 5, size=100)
    test = check_normality(arr)

    # check p-value is > 0.1 for normality tests
    for test_key, values in test.items():
        with suppress(IndexError):  # will skip kurtosis and skewness
            assert test[test_key]["pval"] > 0.1

    # skewness and kurtosis must be close to zero
    assert abs(test["skewness"]) < 0.5
    assert abs(test["kurtosis"]) < 0.5


def test_check_normality_exponential_dist():
    """Test when input array exhibits a non-normal distribution."""
    # create test array and get result
    arr = np.random.exponential(50, size=100)
    test = check_normality(arr)

    # check p-value is < 0.05 for normality tests
    for test_key, values in test.items():
        with suppress(IndexError):  # will skip kurtosis and skewness
            assert test[test_key]["pval"] < 0.05

    # skewness and kurtosis must be higher
    assert abs(test["skewness"]) > 0.1
    assert abs(test["kurtosis"]) > 0.1


def test_check_normality_large_sample_warning():
    """Test warning is raised when sample size is large."""
    arr = np.random.normal(size=3000)
    with pytest.warns(UserWarning, match="sample size is high"):
        check_normality(arr)


def test_check_normality_output_structure():
    """Test all expected keys are returned."""
    arr = np.random.normal(size=100)
    results = check_normality(arr)
    expected_keys = {
        "shapiro_wilk",
        "dagostino_pearson",
        "anderson_darling",
        "skewness",
        "kurtosis",
    }
    assert set(results.keys()) == expected_keys
