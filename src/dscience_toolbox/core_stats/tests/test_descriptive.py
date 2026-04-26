"""Unit tests for functions in descriptive.py."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from dscience_toolbox.core_stats import column_dtypes, summary_stats


def test_column_dtypes(titanic_df):
    """Test columns are identified with the correct dtypes."""
    # get expected number of columns with a dtype and run test function
    exp_count = {"str": 5, "int64": 4, "float64": 2, "bool": 2, "category": 2}
    test = column_dtypes(titanic_df)

    # assertions
    assert all(len(cols) == exp_count[dt] for dt, cols in test.items())
    for _dtype, cols in test.items():
        assert np.all(titanic_df[cols].dtypes == _dtype)


def test_summary_stats_output_columns(titanic_df):
    """Test format of the output DataFrame."""
    # test returned columns are numeric only
    test_df = summary_stats(titanic_df)
    numeric_cols = titanic_df.select_dtypes(include="number").columns
    assert np.all(test_df.columns == numeric_cols)

    # test additional metrics not included in pandas describe() are present
    new_metrics = {"missing", "missing_%", "skewness", "kurtosis", "CV_%"}
    assert new_metrics.issubset(test_df.index)

    # test if output includes only desired column subset
    test_df_subset = summary_stats(titanic_df, subset=["age", "fare"])
    assert sorted(test_df_subset.columns) == ["age", "fare"]


def test_summary_stats_missing_stats(titanic_df):
    """Test if missing value statistics are calculated correctly."""
    # calculate expected and test missing value statistics
    exp_missing = titanic_df["age"].isna().sum()
    exp_missing_perc = 100 * exp_missing / titanic_df.shape[0]
    test_df = summary_stats(titanic_df)

    # create arrays for assertion
    expected = np.array([exp_missing, exp_missing_perc])
    test = test_df.loc[["missing", "missing_%"], "age"].to_numpy()
    assert_allclose(expected, test, atol=0.001)


def test_summary_stats_no_numeric_error(titanic_df):
    """Test that an error is raised when no numeric columns are in the input"""
    input_df = titanic_df.select_dtypes(exclude="number")
    with pytest.raises(ValueError, match="no numeric columns found"):
        summary_stats(input_df)
