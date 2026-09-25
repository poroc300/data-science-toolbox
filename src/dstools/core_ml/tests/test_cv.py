"""Unit tests for functions in cv.py."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from dstools.core_ml import train_val_test_split


@pytest.mark.parametrize("convert_to_array", (True, False))
def test_train_val_test_split_sizes(titanic_df_xy, convert_to_array):
    """Test that the default split produces the expected subset sizes."""
    x_data, y_data = titanic_df_xy
    if convert_to_array:
        x_data, y_data = x_data.to_numpy(), y_data.to_numpy()
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
        *titanic_df_xy,
        random_state=42,
    )
    n_samples = len(titanic_df_xy[0])

    # assertions
    assert len(x_train) == n_samples - int(np.ceil(n_samples * 0.2)) - int(
        np.ceil(n_samples * 0.1)
    )
    assert len(x_val) == int(np.ceil(n_samples * 0.1))
    assert len(x_test) == int(np.ceil(n_samples * 0.2))
    assert len(y_train) == len(x_train)
    assert len(y_val) == len(x_val)
    assert len(y_test) == len(x_test)


def test_train_val_test_split_preserves_alignment(titanic_df_xy):
    """Test that features and targets remain aligned after splitting."""
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
        *titanic_df_xy,
        random_state=42,
    )
    for x_split, y_split in (
        (x_train, y_train),
        (x_val, y_val),
        (x_test, y_test),
    ):
        assert_array_equal(x_split.index.to_numpy(), y_split.index.to_numpy())


def test_train_val_test_split_disjoint_splits(titanic_df_xy):
    """Test that training, validation, and test sets do not overlap."""
    x_train, x_val, x_test, y_train, y_val, y_test = train_val_test_split(
        *titanic_df_xy,
        random_state=42,
    )
    train_indices = set(x_train.index)
    val_indices = set(x_val.index)
    test_indices = set(x_test.index)

    # assertions
    assert train_indices.isdisjoint(val_indices)
    assert train_indices.isdisjoint(test_indices)
    assert val_indices.isdisjoint(test_indices)
    assert len(train_indices | val_indices | test_indices) == titanic_df_xy[0].shape[0]


def test_train_val_test_split_integer_sizes(titanic_df_xy):
    """Test that absolute sample sizes are supported."""
    x_train, x_val, x_test = train_val_test_split(
        titanic_df_xy[0],
        test_size=100,
        val_size=50,
        random_state=42,
    )
    assert x_test.shape[0] == 100
    assert x_val.shape[0] == 50
    assert x_train.shape[0] == titanic_df_xy[0].shape[0] - 150


def test_train_val_test_split_reproducible_split(titanic_df_xy):
    """Test that the same random state produces the same split."""
    split_1 = train_val_test_split(*titanic_df_xy, random_state=42)
    split_2 = train_val_test_split(*titanic_df_xy, random_state=42)
    for first, second in zip(split_1, split_2):
        try:
            assert_frame_equal(first, second)
        except AssertionError:
            assert_series_equal(first, second)


def test_train_val_test_split_no_shuffle(titanic_df_xy):
    """Test that disabling shuffling preserves the original order."""
    x_data = titanic_df_xy[0]
    x_train, x_val, x_test = train_val_test_split(
        x_data,
        test_size=0.2,
        val_size=0.1,
        shuffle=False,
    )

    # expected number of samples in test/validation sets
    n_test = int(np.ceil(x_data.shape[0] * 0.2))
    n_val = int(np.ceil(x_data.shape[0] * 0.1))

    # expected splits
    expected_train = x_data.iloc[n_test + n_val :]
    expected_val = x_data.iloc[n_test : n_test + n_val]
    expected_test = x_data.iloc[:n_test]

    # assertions
    assert_frame_equal(x_train, expected_train)
    assert_frame_equal(x_val, expected_val)
    assert_frame_equal(x_test, expected_test)


@pytest.mark.parametrize(
    "test_size,val_size",
    [
        (0, 0.1),
        (1.0, 0.1),
        (-0.1, 0.1),
        (0.2, 0),
        (0.2, 1.0),
        (0.2, -0.1),
        (0.6, 0.4),  # exhausts training data
        (-1, 200),
        (10_000, 200),
        (500, 500),  # exhausts training data
    ],
)
def test_train_val_test_split_invalid_split_sizes(titanic_df_xy, test_size, val_size):
    """Test that invalid split sizes raise ValueError."""
    with pytest.raises(ValueError):
        train_val_test_split(*titanic_df_xy, test_size=test_size, val_size=val_size)


@pytest.mark.parametrize(
    "test_size,val_size",
    [
        ("0.2", 0.1),
        (0.2, "0.1"),
        (None, 0.1),
        (0.2, None),
    ],
)
def test_train_val_test_split_invalid_size_types(titanic_df_xy, test_size, val_size):
    """Test that invalid size types raise TypeError."""
    with pytest.raises(TypeError):
        train_val_test_split(*titanic_df_xy, test_size=test_size, val_size=val_size)
