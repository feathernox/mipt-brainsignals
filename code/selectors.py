
# coding: utf-8

# In[ ]:

from selector import *
from genetic import *

class ModelScore():
    def __init__(model):
        self.model = model
    def score(X, y):
        return self.model.fit(X, y).score(X, y)

class GeneticSelector(Selector):
    def __init__(self, score_func):
        self.algo = GeneticAlgorithm(score_func)
        self.name = "Genetic " + str(score_func)[:10]
    

class ModelGeneticSelector(GeneticSelector):
    def __init__(self, model):
        self.model_score = ModelScore(model)
        super(ModelGeneticSelector, self).__init__(self.model_score.score)
        self.name = "Genetic, model=" + str(model)[:10]
        
class LinearGeneticSelector(ModelGeneticSelector):
    def __init__(self):
        from sklearn.linear_model import LinearRegression
        super(LinearGeneticSelector, self).__init__(LinearRegression())
        self.name = "LinReg Genetic"


# In[ ]:



