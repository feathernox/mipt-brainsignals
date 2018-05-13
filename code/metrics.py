import numpy as np


def _check_features_mask(a):
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


def _num_samples(X):
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


def _check_consistent_num_samples(*arrays):
    """Check that all arrays have the same number of samples.

    Args:
        *arrays
    Raises:
        ValueError:
    """
    nums_samples = [_num_samples(X) for X in arrays]
    if len(np.unique(nums_samples)) > 1:
        raise


def _check_array(Y):
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


def _check_regression_answers(Y_true, Y_pred):
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


def residual_square_sum(Y_true, Y_pred):
    """Computes
    RSS = (Y_true - Y_pred)^2

    :param Y_true:
    :param Y_pred:
    :return:
    """
    Y_true, Y_pred = _check_regression_answers(Y_true, Y_pred)

    score = ((Y_true - Y_pred) ** 2).sum()

    return score

def total_square_sum(Y_true):
    """

    :param Y_true:
    :return:
    """
    Y_true = _check_array(Y_true)

    score = ((Y_true - Y_true.mean()) ** 2).sum()

    return score


def determination_coefficient(Y_true, Y_pred, adjusted=True):
    """
    Closer R^2_adj to 1, better
    :param Y_true:
    :param Y_pred:
    :param adjusted:
    :return:
    """
    rss = residual_square_sum(Y_true, Y_pred)
    tss = total_square_sum(Y_true)
    #print("RSS", "TSS", rss, tss)
    if adjusted:
        # check Y_pred is 2d
        m, k = Y_true.shape
        #print("M", "K", m, k)
        score = 1 - (rss / tss) * ((m - k) / (m - 1))
    else:
        score = 1 - rss / tss
    #print(score)
    return score


def variance_inflation_factor(Y_true, Y_pred):
    """Calculates variance inflation factor:
    VIF = 1 / (1 - R^2)

    :param Y_true:
    :param Y_pred:
    :return:
    """

    r2 = determination_coefficient(Y_true, Y_pred)
    score = 1 / (1 - r2)

    return score


def mallows_Cp(Y_true, Y_pred, Y_pred_p, p):
    """C_p = RSS_p / RSS - m + 2 * p,
    where m is
    :param Y_true:
    :param Y_pred:
    :param Y_pred_p:
    :param p:
    :return:
    """
    m = Y_true.shape[0]

    # check that p is int or bool array
    rss = residual_square_sum(Y_true, Y_pred)
    rss_p = residual_square_sum(Y_true, Y_pred_p)
    #print(rss, rss_p, ((Y_pred-Y_pred_p)**2).sum())
    score = rss_p / rss - m + 2 * p
    return score


def bayesian_information_criterion(Y_true, Y_pred, p):
    """
    BIC = RSS + p log m
    The smaller value of BIC is the better model fits the target vector.
    Args:

    Returns:

    """
    m = Y_true.shape[0]

    rss = residual_square_sum(Y_true, Y_pred)
    score = rss + p * np.log(m)

    return score


def condition_number_xtx(X):
    """

    :param X:
    :return:
    """
    X = _check_array(X)

    eigenvalues = (np.linalg.svd(X)[1]) ** 2
    eigenvalues = eigenvalues[np.nonzero(eigenvalues)]
    l_max, l_min = np.max(eigenvalues), np.min(eigenvalues)
    score = l_max / l_min

    return score


