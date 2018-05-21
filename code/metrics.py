import numpy as np

from utils import check_2d_array
from utils import check_features_mask
from utils import check_regression_answers


def residual_square_sum(Y_true, Y_pred):
    """Residual square sum regression score function.

    Parameters
    ----------
    Y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    Y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    score : float
        Residual sum of squares.
        A non-negative floating point value (the best value is 0.0).
    """
    Y_true, Y_pred = check_regression_answers(Y_true, Y_pred)

    score = ((Y_true - Y_pred) ** 2).sum()
    return score


def total_square_sum(Y_true):
    """Total square sum score regression score function.
    Defined as sum, over all observations, of the squared differences of each
    observation from the overall mean.

    Parameters
    ----------
    Y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    Returns
    -------
    score : float
        Total sum of squares. A non-negative floating point value.
    """
    Y_true = check_2d_array(Y_true)

    score = ((Y_true - Y_true.mean()) ** 2).sum()
    return score


def determination_coefficient(Y_true, Y_pred, adjusted=False):
    """Coefficient of determination (R^2) regression score function.
    Computed as 1 - RSS / TSS, where RSS is residual square sum
    and TSS is total square sum.

    The adjusted coefficient of determination considers adding
    redundant features and is defined as
    R^2 = 1 - (RSS / (n_samples - n_outputs)) / (TSS / (n_samples - 1)).

    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). A constant model that always
    predicts the expected value of y, disregarding the input features,
    would get a R^2 score of 0.0.

    Parameters
    ----------
    Y_true : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Ground truth (correct) target values.

    Y_pred : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Estimated target values.

    adjusted : boolean, optional (default=False)
        Whether to compute R^2 or adjusted R^2.

    Returns
    -------
    score : float
        Coefficient of determination score.
    """
    Y_true, Y_pred = check_regression_answers(Y_true, Y_pred)

    rss = residual_square_sum(Y_true, Y_pred)
    tss = total_square_sum(Y_true)
    if adjusted:
        m, k = Y_true.shape
        score = 1 - (rss / tss) * ((m - k) / (m - 1))
    else:
        score = 1 - rss / tss
    return score


def variance_inflation_factor(Y_true, Y_pred):
    """Variance inflation factor regression score function.
    Defined as the ratio of variance in a model with multiple terms,
    divided by the variance of a model with one term alone.
    Computed as VIF = 1 / (1 - R^2), where R^2 is coefficient
    of determination.

    Used for severity of multicollinearity in an ordinary least squares
    regression analysis. Y_pred should be computed using linear regression with
    all features except one. Then large values of VIF (by rule of thumb,
    VIF > 5), indicate collinearity of abandoned feature with others.

    Parameters
    ----------
    Y_true : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Ground truth (correct) target values.

    Y_pred : array-like of shape = [n_samples] or [n_samples, n_outputs]
        Estimated target values.

    Returns
    -------
    score : float
        Coefficient of determination score.
    """
    Y_true, Y_pred = check_regression_answers(Y_true, Y_pred)

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
    Y_true, Y_pred = check_regression_answers(Y_true, Y_pred)
    Y_true, Y_pred_p = check_regression_answers(Y_true, Y_pred_p)
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
    X = check_2d_array(X)

    eigenvalues = (np.linalg.svd(X)[1]) ** 2
    eigenvalues = eigenvalues[np.nonzero(eigenvalues)]
    l_max, l_min = np.max(eigenvalues), np.min(eigenvalues)
    score = l_max / l_min

    return score


