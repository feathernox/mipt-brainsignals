import numpy as np

def _check_features_array(A):
    return True

def _check_array(Y):
    return True

def _check_regression_answers(Y_true, Y_pred):
    '''

    :param Y_true:
    :param Y_pred:
    :return:
    '''
    if Y_true.ndim == 1:
        Y_true = Y_true.reshape((-1, 1))

    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape((-1, 1))

    return Y_true, Y_pred


def residual_square_sum(Y_true, Y_pred):
    '''

    :param Y_true:
    :param Y_pred:
    :return:
    '''
    Y_true, Y_pred = _check_regression_answers(Y_true, Y_pred)

    score = ((Y_true - Y_pred) ** 2).sum()

    return score

def total_square_sum(Y_true):
    '''

    :param Y_true:
    :return:
    '''
    Y_true = _check_array(Y)

    score = ((Y_true - Y_true.mean()) ** 2)

    return score


def determination_coefficient(Y_true, Y_pred, adjusted=True):
    '''

    :param Y_true:
    :param Y_pred:
    :param adjusted:
    :return:
    '''
    rss = residual_square_sum(Y_true, Y_pred)
    tss = total_square_sum(Y_true)
    if adjusted:
        # check Y_pred is 2d
        m, k = Y_true.shape
        score = 1 - (rss / tss) * ((m - k) / (m - 1))
    else:
        score = 1 - rss / tss

    return score


def variance_inflation_factor(Y_true, Y_pred):
    '''

    :param Y_true:
    :param Y_pred:
    :return:
    '''

    r2 = determination_coefficient(Y_true, Y_pred)
    score = 1 / (1 - r2)

    return score


def mallowss_Cp(Y_true, Y_pred, Y_pred_p, p):
    '''

    :param Y_true:
    :param Y_pred:
    :param Y_pred_p:
    :param p:
    :return:
    '''
    m = Y_true.shape[0]

    # check that p is int or bool array
    rss = residual_square_sum(Y_true, Y_pred)
    rss_p = residual_square_sum(Y_true, Y_pred_p)

    score = rss_p / rss - m + 2 * p
    return score

def bayesian_information_criterion(Y_true, Y_pred, p):
    '''

    :return:
    '''
    m = Y_true.shape[0]

    rss = residual_square_sum(Y_true, Y_pred)
    score = rss + p * np.log(m)

    return score

def condition_number_xtx(X):
    '''

    :param X:
    :return:
    '''
    X = _check_array(X)

    eigenvalues = (np.linalg.svd(X)[1]) ** 2
    eigenvalues = eigenvalues[np.nonzero(eigenvalues)]
    l_max, l_min = np.max(eigenvalues), np.min(eigenvalues)
    score = l_max / l_min

    return score


