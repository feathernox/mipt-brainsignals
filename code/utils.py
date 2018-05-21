# coding=utf-8
import numpy as np


def _num_samples(x):
    """From scikit-learn 0.19.1 utils.
    Return number of samples in array-like x.
    """
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)


def check_2d_array(X):
    """Check that X is array suitable for regression task
    and transforms it to 2d array, else raises ValueError.

    Parameters
    ----------
    X : 1d or 2d list or numpy array,
        shape = [n_samples] or [n_samples, n_features]

    Returns
    ----------
    X_converted : 2d array, shape = [n_samples, n_features]
        The converted and validated X.
    """
    X = np.array(X, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape((-1, 1))
    elif X.ndim > 2:
        raise ValueError("Too many dimensions: {0}".format(X.ndim))
    elif X.ndim == 0:
        raise ValueError("No dimensions")
    elif X.size == 0:
        raise ValueError("Singleton array %r cannot be considered"
                         " a valid collection." % X)

    return X


def check_consistent_length(*arrays):
    """From scikit-learn 0.19.1 utils.
    Check that all arrays have consistent first dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])


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
    X_converted : 2d array, shape = [n_samples, n_features]
        The converted and validated X.

    Y_converted : 2d array, shape = [n_samples, n_features]
        The converted and validated Y.
    """
    X = check_2d_array(X)
    Y = check_2d_array(Y)
    check_consistent_length(X, Y)
    return X, Y


def check_random_state(seed):
    """From scikit-learn 0.19.1 utils.
    Turn seed into a np.random.RandomState instance

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
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)


def check_regression_answers(Y_true, Y_pred):
    """Сheck that Y_true and Y_pred are 2d real-valued matrices
    of the same shape [n_samples, n_outputs].

    Parameters
    ----------
    Y_true : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Ground truth (correct) target values.

    Y_pred : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Estimated target values.

    Returns
    ----------
    Y_true_converted : 2d array, shape = [n_samples, n_outputs]
        The converted and validated Y_true.

    Y_pred_converted : 2d array, shape = [n_samples, n_outputs]
        The converted and validated Y_pred.
    """
    Y_true = check_2d_array(Y_true)
    Y_pred = check_2d_array(Y_pred)
    if Y_true.shape != Y_pred.shape:
        raise ValueError("Y_true and Y_pred have different shapes"
                         " ({0}!={1})".format(Y_true.shape, Y_pred.shape))
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