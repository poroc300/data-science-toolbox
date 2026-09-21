"""Functions for creating and managing data splits and cross-validation folds."""

import numpy as np


def train_val_test_split(
    x_data,
    y_data=None,
    test_size=0.2,
    val_size=0.1,
    shuffle=True,
    random_state=None,
):
    """
    Split data into training, validation, and test sets.

    The input data is optionally shuffled before being partitioned into three
    mutually exclusive subsets. ``test_size`` and ``val_size`` can be given as
    proportions of the input data or as absolute numbers of samples.

    Parameters
    ----------
    x_data : ndarray or DataFrame
        Input data to split.
    y_data : ndarray, Series or DataFrame, optional
        Target data corresponding to ``x_data``. If None (default), only
        ``x_data`` is split.
    test_size : float or int, default=0.2
        Proportion or number of samples assigned to the test set.
    val_size : float or int, default=0.1
        Proportion or number of samples assigned to the validation set.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    random_state : int, optional
        Seed used to initialize the random number generator. Set to an integer
        for reproducible splits. The default is None.

    Returns
    -------
    x_train : ndarray or DataFrame
        Training data.
    x_val : ndarray or DataFrame
        Validation data.
    x_test : ndarray or DataFrame
        Test data.
    y_train : ndarray, Series or DataFrame, optional
        Training targets. Returned only when ``y_data`` is provided.
    y_val : ndarray, Series or DataFrame, optional
        Validation targets. Returned only when ``y_data`` is provided.
    y_test : ndarray, Series or DataFrame, optional
        Test targets. Returned only when ``y_data`` is provided.
    """
    n_samples = x_data.shape[0]
    if y_data is not None and y_data.shape[0] != n_samples:
        raise ValueError("x_data and y_data must contain the same number of samples")

    # convert proportions or sample counts to the corresponding sizes
    n_sizes = []
    for i, partition_size in enumerate((test_size, val_size)):
        arg = "test_size" if i == 0 else "val_size"
        if isinstance(partition_size, float):
            if not 0 < partition_size < 1:
                raise ValueError(f"{arg} as a float must be between 0 and 1")
            n_sizes.append(int(np.ceil(n_samples * partition_size)))
        elif isinstance(partition_size, int):
            if not 0 < partition_size < n_samples:
                raise ValueError(f"{arg} must be between 1 and n_samples - 1")
            n_sizes.append(partition_size)
        else:
            raise TypeError(f"{arg} must be a float or int.")

    # unpack sample sizes for test and validation sets
    n_test, n_val = n_sizes

    # check there are samples left for training set
    if n_test + n_val >= n_samples:
        raise ValueError(
            "test_size + val_size must leave at least one sample for training"
        )

    # generate and optionally shuffle the sample indices
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    # partition indices into the three subsets
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    # apply the same split to the input data (preserves input type)
    x_train = x_data.iloc[train_idx] if hasattr(x_data, "iloc") else x_data[train_idx]
    x_val = x_data.iloc[val_idx] if hasattr(x_data, "iloc") else x_data[val_idx]
    x_test = x_data.iloc[test_idx] if hasattr(x_data, "iloc") else x_data[test_idx]
    if y_data is None:
        return x_train, x_val, x_test

    # keep targets aligned with their corresponding samples
    y_train = y_data.iloc[train_idx] if hasattr(y_data, "iloc") else y_data[train_idx]
    y_val = y_data.iloc[val_idx] if hasattr(y_data, "iloc") else y_data[val_idx]
    y_test = y_data.iloc[test_idx] if hasattr(y_data, "iloc") else y_data[test_idx]
    return x_train, x_val, x_test, y_train, y_val, y_test
