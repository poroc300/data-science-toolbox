"""Unit tests for functions in multicollinearity.py."""

import numpy as np
import pandas as pd
import pytest

from dscience_utils.statistics import correlation, vif


@pytest.fixture(scope="function")
def sample_df():
    """Fixture to create a test DataFrame."""
    return pd.DataFrame(
        {
            "x1": np.arange(1, 101),
            "x2": np.arange(1, 101) + 0.1,
            "x3": [1, 3, 2, 5, 4] * 20,
            "x4": list("abcdefghij") * 10,
        }
    )


def test_vif(sample_df):
    """Unit tests for function vif."""
    # test categorical columns are ignored and high/low multicollinearity
    test_vif = vif(sample_df)
    assert "x4" not in test_vif.columns
    assert np.all(test_vif[["x1", "x2"]] > 10)
    assert test_vif["x3"].iat[0] < 5

    # test rows with nan are removed
    sample_df.loc[0, "x1"] = np.nan
    test_vif_nan = vif(sample_df)
    assert np.all(test_vif_nan != test_vif)


def test_correlation(sample_df):
    """Unit tests for function correlation."""
    # test for invalid method
    with pytest.raises(ValueError, match="invalid method supplied"):
        correlation(sample_df, method="invalid")

    # test high/low correlation levels
    results = correlation(sample_df, method="spearman")
    assert results["spearman"].loc["x1", "x2"] >= 0.99
    assert results["spearman"].loc["x1", "x3"] < 0.1
    assert results["spearman_pvalues"].loc["x1", "x2"] < 0.05
    assert results["spearman_pvalues"].loc["x1", "x3"] > 0.5

    # test all keys are returned when method="all"
    assert len(correlation(sample_df, method="all").keys()) == 6
