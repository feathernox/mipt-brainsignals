from sklearn.linear_model import LinearRegression
from genetic import GeneticAlgorithm


class Selector:
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


class ModelScore:
    def __init__(self, model):
        self.model = model

    def score(self, X, y):
        return self.model.fit(X, y).score(X, y)


class GeneticSelector(Selector):
    def __init__(self, score_func, mutation_rate = 0.001,
                 num_iterations = 1000, generation_size = 100):
        self.algo = GeneticAlgorithm(score_func, mutation_rate,
                                     num_iterations, generation_size)
        self.name = "Genetic " + str(score_func)[:10]
    

class ModelGeneticSelector(GeneticSelector):
    def __init__(self, model, mutation_rate = 0.001,
                 num_iterations = 1000, generation_size = 100):
        super(ModelGeneticSelector, self).__init__(model,
                             mutation_rate, num_iterations, generation_size)
        self.name = "Genetic, model=" + str(model)[:10]


class LinearGeneticSelector(ModelGeneticSelector):
    def __init__(self, mutation_rate = 0.001,
                 num_iterations = 1000, generation_size = 100):
        super(LinearGeneticSelector, self).__init__(LinearRegression(),
                        mutation_rate, num_iterations, generation_size)
        self.name = "LinReg Genetic"
