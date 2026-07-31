"""Fixtures used in unit tests."""

import numpy as np
import pandas as pd
import pytest
import seaborn as sns


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


@pytest.fixture(scope="module")
def titanic_df():
    """Fixture to load the Titanic dataset."""
    return sns.load_dataset("titanic")
