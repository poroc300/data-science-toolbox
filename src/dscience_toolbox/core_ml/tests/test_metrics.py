"""Unit tests for functions in metrics.py."""

import numpy as np
import pytest

from dscience_toolbox.core_ml import percent_error, r2_score_adj


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


def test_r2_score_adj_high_low():
    """Test good and bad fits for R2."""
    # create test arrays
    y_true = np.array([1, 2, 3, 4])
    y_pred = y_true.copy()
    
    # calculate good and bad fit R2
    test_high = r2_score_adj(y_true, y_pred, n_predictors=1)
    test_low = r2_score_adj(y_true, y_pred[::-1], n_predictors=1)
    
    # assertions
    assert test_high > 0.99
    assert test_low < 0.01    
  

def test_r2_score_adj_low_sample_size():
    """Test error is raised when there are more predictors than samples."""
    msg = r"sample size must be greater than n_predictors \+ 1"
    with pytest.raises(ValueError, match=msg):
        r2_score_adj(np.ones(3), np.ones(3), n_predictors=10)
        

def test_r2_score_adj_multioutput():
    """Test number of returned R2 values match number of columns of input."""
    arr = np.arange(1, 101).reshape(25, 4)
    test = r2_score_adj(arr, arr, n_predictors=3)
    assert test.size == arr.shape[1]
    
    
    