"""Unit tests for functions in resampling.py."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from dstools.core_stats import bootstrap


@pytest.mark.parametrize("n", (1, 100))
def test_bootstrap_returns_expected_shape(n):
    """Test function returns a scalar per bootstrap replicate."""
    estimates = bootstrap(np.mean, np.arange(n), n_boot=100, random_state=42)
    assert estimates.shape == (100,)


def test_bootstrap_supports_vector_valued_statistics():
    """Test the function supports statistic functions that output arrays."""
    func = lambda x: np.outer(x[:2], x[:3])
    estimates = bootstrap(func, np.arange(10), n_boot=100, random_state=42)
    assert estimates.shape == (100, 2, 3)


def test_bootstrap_resamples_with_replacement():
    """Test function performs sampling with replacement."""
    estimates = bootstrap(np.mean, np.arange(3), n_boot=100, random_state=42)
    assert np.any(estimates != 1)


def test_bootstrap_supports_multiple_samples(ttest_arrays):
    """Test function accepts multiple samples in the input."""
    diff_means = lambda x, y: np.mean(x) - np.mean(y)
    sample1, sample2 = ttest_arrays["arr1"], ttest_arrays["arr2"]
    estimates = bootstrap(diff_means, sample1, sample2, n_boot=100, random_state=42)
    assert estimates.shape == (100,)


def test_bootstrap_samples_are_resampled_independently():
    """Test that different samples are involve in the calculation of the bootstrapped statistic."""
    sample1 = np.array([1, 2, 3])
    sample2 = np.array([10, 20, 30])
    add_means = lambda x,y: np.mean(x) + np.mean(y)
    estimates = bootstrap(add_means, sample1, sample2, n_boot=100, random_state=42)

    # check both samples participate in the statistic
    assert np.all(estimates > 1)
    assert np.all(estimates < 31)


def test_bootstrap_is_reproducible():
    """Test supplying same seed ensures equal results."""
    sample = np.arange(10)
    estimates1 = bootstrap(np.mean, sample, n_boot=100, random_state=42)
    estimates2 = bootstrap(np.mean, sample, n_boot=100, random_state=42)
    assert_array_equal(estimates1, estimates2)


def test_bootstrap_requires_at_least_one_sample():
    """Test function raises ValueError if no sample is provide."""
    with pytest.raises(ValueError, match="at least one sample"):
        bootstrap(np.mean, n_boot=100)


@pytest.mark.parametrize("n_boot", [1.2, 0, -1, "dummy"])
def test_bootstrap_rejects_non_positive_n_boot(n_boot):
    """Test function raises ValueError if n_boot is not a positive int."""
    sample = np.arange(10)
    with pytest.raises(ValueError, match="positive integer"):
        bootstrap(np.mean, sample, n_boot=n_boot)
