"""Unit tests for functions in _validators.py."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from dstools.utils._validators import (
    _check_plotting_inputs,
    _merge_plot_kwargs,
    _validate_numeric_array,
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


def test_validate_numeric_array_dataframe(titanic_df):
    """Test non-numeric columns are filtered out."""
    with pytest.warns(UserWarning, match="dropping non-numeric"):
        result = _validate_numeric_array(titanic_df)
        assert titanic_df.shape != result.shape


def test_validate_numeric_array_dataframe_not_numeric(titanic_df):
    """Test that ValueError is raised if input DataFrame does not have numeric columns."""
    with pytest.raises(ValueError, match="has no numeric columns"):
        _validate_numeric_array(titanic_df[["sex", "alive"]])


def test_validate_numeric_array_dataframe_numeric_only(titanic_df):
    """Test output DataFrame is equal to input if it contains only numeric columns."""
    expected = titanic_df[["fare", "age"]].copy()
    result = _validate_numeric_array(expected)
    assert_frame_equal(expected, result)


@pytest.mark.parametrize("arr", ([1, 2], (1, 2)))
def test_validate_numeric_array_native_sequences(arr):
    """Test lists or tuples with numbers only are coerced to numpy arrays."""
    assert isinstance(_validate_numeric_array(arr), np.ndarray)


@pytest.mark.parametrize(
    "arr", ([1, "2"], (1, "2"), np.array([1, "2"]), pd.Series([1, "2"]))
)
def test_validate_numeric_array_not_numeric(arr):
    """Test non-numeric arrays raise ValueError."""
    with pytest.raises(ValueError, match="does not have a numeric dtype"):
        _validate_numeric_array(arr)


@pytest.mark.parametrize("arr", (tuple(), [], np.array([]), pd.Series([])))
def test_validate_numeric_array_empty(arr):
    """Test arrays without values raise ValueError."""
    with pytest.raises(ValueError, match="an empty sequence was supplied"):
        _validate_numeric_array(arr)


@pytest.mark.parametrize("arr", (1, {1: 2}, "dummy"))
def test_validate_numeric_array_invalid_dtype(arr):
    """Test TypeError is raised with invalid input types."""
    with pytest.raises(TypeError):
        _validate_numeric_array(arr)


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
