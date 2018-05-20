import numpy as np


from utils import check_X_Y
from utils import check_random_state


def bootstrap(X, Y,
              n_datasets=1,
              n_samples=None,
              random_state=None):
    """Performs bootstrap procedure:
    using original training and target matrices, returns `n_datasets`
    pairs of new training and target matrices, which have
    `len_sample` rows of corresponding features from training matrix and
    outputs from target matrix, numbers of rows are chosen randomly.

    Parameters
    ----------
    X : array-like, shape = [n_samples], [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of predictors.

    Y : array-like, shape = [n_samples] or [n_samples, n_outputs]
        Target vectors, where n_samples is the number of samples and
        n_targets is the number of response variables.

    n_datasets : int, optional (default=1)
        Number of samples.

    n_samples : int or None, optional (default=None)
        Length of samples. If None, n_samples is the same as X and Y.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    sample_X : array of shape = [n_datasets, n_samples, n_features]
        `n_datasets` bootstrapped training matrices.

    sample_Y : array of shape = [n_datasets, n_samples, n_outputs]
        `n_datasets` bootstrapped target matrices.
    """
    X, Y = check_X_Y(X, Y)
    random_state = check_random_state(random_state)
    dataset_n_samples = n_samples

    if dataset_n_samples is None:
        dataset_n_samples = X.shape[0]

    n_samples = X.shape[0]
    indices = random_state.choice(n_samples,
        (n_datasets, dataset_n_samples), replace=True)

    X_sample = X[indices, :]
    Y_sample = Y[indices, :]
    return X_sample, Y_sample
