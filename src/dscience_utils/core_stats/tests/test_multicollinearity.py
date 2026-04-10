"""Unit tests for functions in multicollinearity.py."""

import numpy as np
import pytest

from dscience_utils.core_stats import correlation, vif


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


def test_correlation_invalid_method(sample_df):
    """Test error is raised when invalid method name is supplied."""
    with pytest.raises(ValueError, match="invalid method supplied"):
        correlation(sample_df, method="invalid")
