"""Unit tests for functions in _validators.py."""

import pandas as pd
import pytest

from dstools.utils import (
    _check_plotting_inputs,
    _merge_plot_kwargs,
    _validate_pandas_dataframe,
)


@pytest.mark.parametrize("df", (pd.Series([1, 2]), pd.DataFrame([[2, 2]])))
def test_validate_pandas_dataframe(df):
    """Function with an input as DataFrame or Series."""
    df = _validate_pandas_dataframe(df, input_arg="arg")
    assert isinstance(df, pd.DataFrame)


def test_validate_pandas_dataframe_invalid_input():
    """Test invalid input."""
    msg = "arg must be a pandas DataFrame or Series, got list"
    with pytest.raises(TypeError, match=msg):
        _validate_pandas_dataframe([3], input_arg="arg")


def test_check_plotting_inputs(sample_df):
    """Test the function _check_plotting_inputs."""
    # columns exist in DataFrame
    _check_plotting_inputs(sample_df, {"x1", "x2"})

    # at least one column does not exist
    msg = "missing columns in DataFrame: {'dummy'}"
    with pytest.raises(KeyError, match=msg):
        _check_plotting_inputs(sample_df, {"x1", "x2", "dummy"})


@pytest.mark.parametrize(
    "params, kwargs, expected",
    (
        ({"a": 2}, {"b": 10}, {"a": 2, "b": 10}),
        ({}, {"b": 10}, {"b": 10}),
        ({"a": 2}, {"a": 5, "b": 10}, {"a": 2, "b": 10}),
    ),
)
def test_merge_plot_kwargs(params, kwargs, expected):
    """Test the function _merge_plot_kwargs."""
    test = _merge_plot_kwargs(params, kwargs)
    assert expected == test


def test_merge_plot_kwargs_warning_duplicated():
    """Test the function _merge_plot_kwargs."""
    with pytest.warns(UserWarning, match="ignoring values in plot_kwargs"):
        _merge_plot_kwargs({"a": 2}, {"a": 5, "b": 10})
