class Selector():
    '''Wrapper over a selection algorithm'''
    def __init__(self, algo, name=None):
        self.algo = algo
        self.name = name
    def fit(self, X, y):
        self.algo.fit(X, y)
    def select(self, num_features):
        return self.algo.select(num_features)
    def __str__(self):
        return self.name






