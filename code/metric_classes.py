from metrics import *

print(condition_number_xtx)

class FunctionEvaluator():
    def __init__(self, func=None):
        if func == None:
            self.function = lambda a, b: 0
        else:
            self.function = func

class Metric(FunctionEvaluator):
    def evaluate(self, model, X, y):
        z = self.function(y, model.predict(X))
        return z
    

class Characteristic(FunctionEvaluator):
    def evaluate(self, model):
        pass
    

class Comparison(FunctionEvaluator):
    def evaluate(self, full_model, model, full_X, y, mask):
        pass

class RSS(Metric):
    def __init__(self):
        super(RSS, self).__init__(residual_square_sum)
    def __str__(self):
        return "RSS"

class Adjusted_Determination_Coefficient(Metric):
    def __init__(self):
        super(Adjusted_Determination_Coefficient, self).__init__(determination_coefficient)
    def __str__(self):
        return "Adj Det Coef"
        
class Determination_Coefficient(Metric):
    def __init__(self):
        super(Determination_Coefficient, self).__init__(determination_coefficient)
    def evaluate(self, model, X, y):
        z = self.function(y, model.predict(X), False)
        #print("Z", z)
        return z
    def __str__(self):
        return "Det Coef"

class VIF(Metric):
    def __init__(self):
        super(VIF, self).__init__(variance_inflation_factor)
    def __str__(self):
        return "VIF"

class BIC(Metric):
    def __init__(self):
        super(BIC, self).__init__(bayesian_information_criterion)
    def evaluate(self, model, X, y):
        num_features = X.shape[1]
        z = self.function(y, model.predict(X), num_features)
        #print("Z", z)
        return z
    def __str__(self):
        return "BIC"

class Cp(Comparison):
    def __init__(self):
        super(Cp, self).__init__(mallows_Cp)
    def evaluate(self, full_model, model, full_X, reduced_X, y):
        y_full = full_model.predict(full_X)
        y_p = model.predict(reduced_X)
        p = reduced_X.shape[1]
        z = self.function(y, y_full, y_p, p)
        return z
    def __str__(self):
        return "Mallow's Cp"
        
        

class Condition_Number(Characteristic):
    def __init__(self):
        super(Condition_Number, self).__init__(condition_number_xtx)
    def evaluate(self, model):
        X = model.coef_
        return self.function(X)
    def __str__(self):
        return "Cond number"


# In[ ]:



