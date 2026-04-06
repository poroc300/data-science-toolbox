"""Unit tests for functions in multicollinearity.py."""

import numpy as np
import pandas as pd
import pytest

from dscience_utils.statistics import vif


@pytest.fixture(scope="function")
def sample_df():
    """Fixture to create a test DataFrame."""
    return pd.DataFrame(
        {
            "x1": np.arange(1, 11),
            "x2": np.arange(1, 11) + 0.1,
            "x3": [1, 3, 2, 5, 4] * 2,
            "x4": list("abcdefghij"),
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
