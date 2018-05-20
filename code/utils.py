# coding=utf-8
import numpy as np


def check_2d_array(X):
    """Check that X is array suitable for regression task
    and transforms it to 2d array, else raises ValueError.

    Parameters
    ----------
    X : 1d or 2d list or numpy array,
        shape = [n_samples] or [n_samples, n_features]

    Returns
    ----------
    X_converted : transformed arra
    """
    X = np.array(X)

    if X.ndim == 1:
        X = X.reshape((-1, 1))
    elif X.ndim > 2:
        raise ValueError("Too many dimensions: {0}".format(X.ndim))
    elif X.ndim == 0:
        raise ValueError("No dimensions")
    elif X.size == 0:
        raise ValueError("No elements")

    return X


def check_consistent_lengths(X, Y):
    """Check that both arrays have the same number of samples.

    Parameters
    ----------
    X : 1d or 2d array, shape = [n_samples] or [n_samples, n_features]
        Training data.

    Y : 1d or 2d array, shape = [n_samples] or [n_samples, n_features]
        Target data.


    """
    pass


def check_X_Y(X, Y):
    """Сheck that X and Y are real-valued matrices of shapes
    [n_samples, n_features] and [n_samples, n_outputs].

    Parameters
    ----------
    X : array-like, shape = [n_samples] or [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of predictors.

    Y : array-like, shape = [n_samples] or [n_samples, n_outputs]
        Target vectors, where n_samples is the number of samples and
        n_targets is the number of response variables.

    Returns
    ----------
    X : array-like, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of predictors.

    Y : array-like, shape = [n_samples] or [n_samples, n_outputs]
        Target vectors, where n_samples is the number of samples and
        n_targets is the number of response variables.
    """
    converted_X = check_2d_array(X)
    converted_Y = check_2d_array(Y)
    check_consistent_lengths(converted_X, converted_Y)
    return converted_X, converted_Y



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