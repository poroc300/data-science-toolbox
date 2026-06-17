"""Unit tests for functions in arithmetic.py."""

import numpy as np
import pandas as pd
import pytest

from dstools.maths import rceil, rfloor


def test_rceil_decimals():
    """Test with 2 decimals of precision."""
    arr = np.array([1.231, 2.8769])
    expected = np.array([1.24, 2.88])
    test = rceil(arr, dec=2)
    assert np.all(expected == test)


@pytest.mark.parametrize("dec", (None, 0))
def test_rceil_matches_npceil(dec):
    """Test dec=None or dec=0 matches np.ceil behaviour."""
    arr = np.array([1.2, 2.8, -1.2])
    expected = np.ceil(arr)
    test = rceil(arr, dec=dec)
    assert np.all(expected == test)


@pytest.mark.parametrize(
    "arr, expected",
    (
        (pd.Series([1.23, 2.82]), np.array([1.3, 2.9])),
        ([1.23, 2.82], np.array([1.3, 2.9])),
        (pd.DataFrame(np.full((3, 3), 1.23)), np.full((3, 3), 1.3)),
        (1.23, 1.3),
    ),
)
def test_rceil_input_types(arr, expected):
    """Test dec=None or dec=0 matches np.ceil behaviour."""
    test = rceil(arr, dec=1)
    assert np.all(expected == test)


@pytest.mark.parametrize("dec", (-1, "a", 1.2))
def test_rceil_invalid_dec(dec):
    """Test for invalid dec values."""
    with pytest.raises(TypeError, match="dec must be None or a positive int"):
        rceil([12.3, 45.6], dec=dec)


def test_rfloor_decimals():
    """Test with 2 decimals of precision."""
    arr = np.array([1.231, 2.8769])
    expected = np.array([1.23, 2.87])
    test = rfloor(arr, dec=2)
    assert np.all(expected == test)


@pytest.mark.parametrize("dec", (None, 0))
def test_rfloor_matches_npfloor(dec):
    """Test dec=None or dec=0 matches np.floor behaviour."""
    arr = np.array([1.2, 2.8, -1.2])
    expected = np.floor(arr)
    test = rfloor(arr, dec=dec)
    assert np.all(expected == test)


@pytest.mark.parametrize(
    "arr, expected",
    (
        (pd.Series([1.23, 2.82]), np.array([1.2, 2.8])),
        ([1.23, 2.82], np.array([1.2, 2.8])),
        (pd.DataFrame(np.full((3, 3), 1.23)), np.full((3, 3), 1.2)),
        (1.23, 1.2),
    ),
)
def test_rfloor_input_types(arr, expected):
    """Test dec=None or dec=0 matches np.ceil behaviour."""
    test = rfloor(arr, dec=1)
    assert np.all(expected == test)


@pytest.mark.parametrize("dec", (-1, "a", 1.2))
def test_rfloor_invalid_dec(dec):
    """Test for invalid dec values."""
    with pytest.raises(TypeError, match="dec must be None or a positive int"):
        rfloor([12.3, 45.6], dec=dec)
