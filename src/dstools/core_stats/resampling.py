"""Functions that apply statistical resampling methods."""

import numpy as np


def bootstrap(statistic, *samples, n_boot=10_000, random_state=None):
    """
    Generate bootstrap replicates of a statistic from one or more samples.

    Parameters
    ----------
    statistic : callable
        Function applied to each bootstrap sample. It must accept one argument
        for each sample and return a scalar or array-like statistic.
    *samples : array-like
        One or more observed samples. Each sample is resampled independently
        with replacement, preserving its original size.
    n_boot : int, default=10_000
        Number of bootstrap replicates to generate.
    random_state : int or numpy.random.Generator, optional
        Seed or random number generator used to generate bootstrap samples.

    Returns
    -------
    estimates : numpy.ndarray
        Bootstrap replicates of `statistic`. The first dimension corresponds
        to bootstrap replicates, while remaining dimensions match the shape of the
        statistic returned by `statistic`.

    Examples
    --------
    import numpy as np
    from dstools.core_stats import bootstrap

    # define statistic function
    diff_means = lambda x,y: np.mean(x) - np.mean(y)

    # generate input arrays
    arr1 = np.random.normal(100, 10, size=100)
    arr2 = np.random.normal(150, 5, size=100)

    # get bootstrapped estimates of the difference of means
    estimates = bootstrap(diff_means, arr1, arr2, n_boot=1000, random_state=42)

    # estimate 95% confidence interval of the difference of means
    ci95 = np.quantile(estimates, [0.025, 0.975])
    """
    # input checks
    if not samples:
        raise ValueError("at least one sample must be provided")
    if not isinstance(n_boot, int) or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")

    # coerce input samples to ndarray
    rng = np.random.default_rng(random_state)
    samples = [np.asarray(sample) for sample in samples]

    # create the first replicate separately to determine the shape of the
    # statistic's output (support for scalar and arrays)
    first_samples = [
        rng.choice(sample, size=len(sample), replace=True) for sample in samples
    ]
    first_estimate = np.asarray(statistic(*first_samples))

    # the first axis indexes bootstrap replicates, any remaining axes
    # preserve the shape of the statistic's output.
    estimates = np.empty((n_boot,) + first_estimate.shape)
    estimates[0] = first_estimate

    # perform bootstrap for remaining replicates
    for i in range(1, n_boot):
        boot_samples = [
            rng.choice(sample, size=len(sample), replace=True) for sample in samples
        ]
        estimates[i] = statistic(*boot_samples)

    return estimates
