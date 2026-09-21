"""Fixtures used in unit tests in core_ml submodule."""

import pytest


@pytest.fixture(scope="module")
def titanic_df_xy(titanic_df):
    """Fixture that splits titanic dataset into feature and target sets."""
    x_data = titanic_df.drop(columns="survived")
    y_data = titanic_df["survived"]
    return x_data, y_data
