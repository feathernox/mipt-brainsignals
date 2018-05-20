import numpy as np


def check_features_mask(a):
    """Check that a is mask suitable for feature selection
    and transforms it to 1d boolean array.

    Args:
        a: 1d array containing zeros and ones.
            Its length corresponds to number of features;
            `1` on i-th position indicates selection of i-th feature,
            `0` -- its abundance.
    Returns:
        a: 1d boolean numpy array.
    Raises:
        ValueError:
            If `a` cannot be transformed to array, its dimensionality is not 1
            or it contains not only zeros and ones.
    """
    a = np.array(a)

    if a.ndim != 1:
        raise ValueError("Dimension of array is {0}"
                         "instead of 1.".format(a.ndim))

    if not np.array_equal(a, a.astype(bool)):
        raise ValueError("Not all array elements are 0 or 1.")

    return a.astype(bool)


def num_samples(X):
    """

    Args:
        X a
    Returns:
        n -- number od
    """
    X = np.array(X)
    if X.ndim == 0:
        raise
    n = X.shape[0]


def check_consistent_lengths(X, Y):
    """Check that all arrays have the same number of samples.

    Args:
        *arrays
    Raises:
        ValueError:
    """
    nums_samples = [_num_samples(X) for X in arrays]
    if len(np.unique(nums_samples)) > 1:
        raise


def check_2d_array(Y):
    """Check that Y is array suitable for regression task
    and transforms it to 2d float array.

    It should be possible to transform Y to 2d array

    Args:
        Y
    Returns:
    Raises:
    :param Y: array-like of shape = (n_samples) or (n_samples, n_outputs)
    :return: Y
    """
    Y = np.array(Y)

    if Y.ndim == 1:
        Y = Y.reshape((-1, 1))

    return Y

def check_X_Y(X, Y):
    """

    :param X:
    :param Y:
    :return:
    """
    X = check_2d_array(X)
    Y = check_2d_array(Y)
    check_consistent_lengths(X, Y)
    return X, Y

def check_regression_answers(Y_true, Y_pred):
    """

    :param Y_true:
    :param Y_pred:
    :return:
    """
    if Y_true.ndim == 1:
        Y_true = Y_true.reshape((-1, 1))

    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape((-1, 1))

    return Y_true, Y_pred

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
