"""Unit tests for functions in _validators.py."""

import numpy as np
import pandas as pd
import pytest

from dscience_toolbox.utils import _to_numpy, _validate_pandas_dataframe


@pytest.mark.parametrize("arr", ([], pd.Series(), np.ones((3, 3, 3))))
def test_to_numpy_invalid_inputs(arr):
    """Test if error is raised on invalid input."""
    with pytest.raises(ValueError):
        _to_numpy(arr)


def test_to_numpy_numbers():
    """Test when input are int or float."""
    # create test arrays
    arr1 = _to_numpy(2.3)
    arr2 = _to_numpy(2, dtype=np.int8, ndim=1)
    arr3 = _to_numpy(2, dtype=np.int8, ndim=2)

    # assertions
    assert arr1 == np.array(2.3) and arr1.dtype == "float64"
    assert np.all(arr2 == np.array([2])) and arr2.dtype == "int8"
    assert np.all(arr3 == np.array([[2]])) and arr2.dtype == "int8"


@pytest.mark.parametrize("arr", ([1, 2], pd.Series([1, 2]), pd.DataFrame([[2, 2]])))
def test_to_numpy_array_like(arr):
    """Test different array-like inputs are coerced to numpy arrays."""
    assert isinstance(_to_numpy(arr), np.ndarray)


def test_to_numpy_ndims():
    """Test change of array dimensions."""
    # create test array
    arr_1d = np.ones(3)
    arr_2d = arr_1d.reshape(-1, 1).copy()

    # assertions
    assert _to_numpy(arr_1d, ndim=2).ndim == 2
    assert _to_numpy(arr_2d, ndim=1).ndim == 1


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
