"""Unit tests for functions in metrics.py."""

import numpy as np
import pytest

from dscience_toolbox.core_ml import percent_error


def test_percent_error_scalar():
    """Test when input has a scalar."""
    test = percent_error(100, np.array([90.0, 100.0, 110.0]))
    assert np.all(test == np.array([-10, 0, 10]))


@pytest.mark.parametrize("abs_", (False, True))
def test_percent_error_arrays(abs_):
    """Test when input are two arrays of same shape."""
    # expected values
    expected = np.repeat([-10, 0, 10], repeats=2).reshape(3, 2)
    if abs_:
        expected = np.abs(expected)

    # run test function and assertions
    true = np.full((3, 2), 100)
    pred = np.repeat([90.0, 100.0, 110.0], repeats=2).reshape(3, 2)
    test = percent_error(true, pred, absolute=abs_)
    assert np.all(expected == test)


def test_percent_error_array_broadcasting():
    """Test when arrays don't have same shape but are broadcastable."""
    # expected values
    expected = np.repeat([-10, 0, 10], repeats=2).reshape(3, 2)

    # run test function and assertions
    true = np.full(3, 100).reshape(-1, 1)
    pred = np.repeat([90.0, 100.0, 110.0], repeats=2).reshape(3, 2)
    test = percent_error(true, pred)
    assert np.all(expected == test)
