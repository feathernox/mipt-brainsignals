
# coding: utf-8

# In[1]:

class Selector():
    '''Wrapper over a selection algorithm'''
    def __init__(self, algo):
        self.algo = algo
    def fit(self, X, y):
        self.algo.fit(X, y)
    def select(self, num_features):
        return self.algo.select(num_features)


# In[ ]:



