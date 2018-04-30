import numpy as np

class Bootstrap():
    def __init__(self, X, y, n_samples, len_sample):
        self.n_samples = n_samples
        self.len_sample = len_sample
        self.X = np.array(X)
        self.y = np.array(y)
        self.sample_X = np.zeros((n_samples, len_sample, self.X.shape[1]))
        self.sample_y = np.zeros((n_samples, len_sample, self.y.shape[1]))
        for i in range(n_samples):
            idx = np.random.choice(np.arange(len(X)), len_sample, replace=True)
            self.sample_X[i] = self.X[idx]
            self.sample_y[i] = self.y[idx]

    def values(self):
        return self.sample_X, self.sample_y

