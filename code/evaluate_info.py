import numpy as np
from metrics import *
from bootstrap import *
from metric_classes import *

class EvaluateInfo():
    '''
    Abstract class.
    Use Static EvaluateInfo or DynamicEvaluateInfo.
    '''
    def __init__(self, model, metrics = [], comparisons = [], characteristics = []):
        '''
        Creates an EvaluateInfo object. This object is used to calculate the quality of a model 
        and the distribution of the quality using bootstrap.
        model - the model to use
        metrics - desired metrics are objects of class Metric or with the same interface(see Metric)
        comparisons - desired comparisons are objects of class Comparison or with the same interface(see Comparison)
        characteristics - desired characteristics are objects of class Сharacteristic
        or with the same interface(see Characteristic)
        '''
        self.model = model
        self.characteristics = characteristics # characteristic of the fitted model itself
        self.metrics = metrics # compares y and y_pred
        self.comparisons = comparisons #compares y, y_pred, y_pred_with_reduced_features
        
        all_names = [str(el) for el in metrics] + [str(el) for el in comparisons] + [str(el) for el in characteristics]
        
        self.names = dict(zip(all_names, [i for i in range(len(all_names))]))
        #print(self.names)
        
    def _calcQuality(self):
        '''
        Calculates the value of all metrics on test sample
        '''
        #print(self.X_train.shape, self.y_train.shape, self.X_test.shape, self.y_test.shape)
        #print(self.masks.shape)
        
        model = self.model
        for (m, mask) in enumerate(self.masks):
            reduced_X_test = self.X_test[:, mask]
            #print(self.X_train.T[mask].T.shape, self.y_train.shape)
            model.fit(self.X_train.T[mask].T, self.y_train)
            for (i, metric) in enumerate(self.metrics):
                self.quality[i][m] = metric.evaluate(model, reduced_X_test, self.y_test)

            for (i, comp) in enumerate(self.comparisons):
                ind = i + len(self.metrics)
                self.quality[ind][m] = comp.evaluate(self.full, model, self.X_test,
                                                                 reduced_X_test, self.y_test)
            for (i, char) in enumerate(self.characteristics):
                ind = i + len(self.metrics) + len(self.comparisons)
                self.quality[ind][m] = char.evaluate(model)
        
    def fit(self, X_train, y_train, X_test, y_test, masks = None, n_samples=20, len_sample=None, boot=None):
        '''
        X_train - train features. 2D numpy array or list
        
        y_train - train answers. 2D numpy array or list
        
        X_test - test features. 2D numpy array or list
        
        y_test - test answers. 2D numpy array or list
        
        masks - sets of desired features. List or numpy array of lists or numpy arays. 
        Each list/numpy array represents a boolean mask of taken features.
        If None, one mask containing all features will be used.
        
        n_samples - number of bootstrap samples to generate. Default is 20.
        
        len_sample - length of each bootstrap sample
        If None, it will be equal to the number of features.
        '''
        
        self.len_sample = len_sample
        
        self.n_samples = n_samples

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        
        if masks is None:
            masks = np.ones((1, self.X_train.shape[1]), dtype=bool)
        
        self.masks = np.array(masks, dtype=bool)
        
        # model with all features
        self.full = deepcopy(self.model.fit(self.X_train, self.y_train))
        
        self.result = np.zeros((len(self.metrics) + len(self.comparisons) 
                                + len(self.characteristics), len(self.masks), self.n_samples))
        
        self.quality = np.zeros((len(self.metrics) + len(self.comparisons) 
                                + len(self.characteristics), len(self.masks)))
        self._calcQuality()
        
    
    
    def get_quality(self):
        '''Returns the value of metrics'''
        return self.quality
    
    def get_result(self):
        '''Returns the value of metrics generated with bootstrap'''
        return self.result
    
    def _get_mapped(self, array):
        '''Returns the value of something mapped to names'''
        return dict(zip(self.names.keys(), list(array)))
    
    def get_mapped_quality(self):
        '''Returns the value of metrics mapped to their names'''
        return self._get_mapped(self.quality)
    
    def get_mapped_result(self):
        '''Returns the value of metrics generated with bootstrap mapped to their names'''
        return self._get_mapped(self.result)
    
    def _by_name(self, name, array):
        '''Return element of array corresponding to the name'''
        if name in self.names:
            return array[self.names[name]]
        else:
            raise NameError('Invalid argument')
        
    def quality_by_name(self, name):
        '''Return quality of a particular metric'''
        return self._by_name(name, self.quality)
            
    def result_by_name(self, name):
        '''Return bootstrap quality of a particular metric'''
        return self._by_name(name, self.result)
            
    

from copy import deepcopy
class EvaluateStaticInfo(EvaluateInfo):
    '''
    Computes the values and distributions of metrics using bootstrap.
    A model is fit on one train sample, but the quality is evaluated on bootstrap test samples
    '''
    def __init__(self, model, metrics = [], comparisons = []):
        super(EvaluateStaticInfo, self).__init__(model, metrics, comparisons, [])
    
    def __eval(self):
        '''
        Computes the bootstrap result
        '''
        model = self.model
        
        
        sample_X, sample_y = self.boot.values()
        
        #print(sample_X.shape, sample_y.shape)
        
        self.models = []
        for mask in self.masks:
            self.models += [deepcopy(model.fit(self.X_train[:, mask], self.y_train))]
        
        for (m, mask) in enumerate(self.masks):
            for it in range(self.n_samples):
                reduced_X_cur = (sample_X[it].T[mask]).T
                for (i, metric) in enumerate(self.metrics):
                    self.result[i][m][it] = metric.evaluate(self.models[m], reduced_X_cur, sample_y[it])
                for(i, comp) in enumerate(self.comparisons):
                    ind = i + len(self.metrics)
                    self.result[ind][m][it] = comp.evaluate(self.full, self.models[m], sample_X[it], 
                                                              reduced_X_cur, sample_y[it])

    def fit(self, X_train, y_train, X_test, y_test, masks = None, n_samples=20, len_sample=None, boot=None):
        '''See EvaluateInfo - fit'''
        super(EvaluateStaticInfo, self).fit(X_train, y_train, X_test, y_test, masks, n_samples, len_sample)
        if self.len_sample is None:
            self.len_sample = len(X_test)
        if boot is None:
            self.boot = Bootstrap(self.X_test, self.y_test, self.n_samples, self.len_sample)
        else:
            self.boot = boot
        self.__eval()
        

from copy import deepcopy
class EvaluateDynamicInfo(EvaluateInfo):
    '''
    Computes the values and distributions of metrics using bootstrap.
    A model is fit on bootstrap train samples, but the quality is evaluated on one test sample
    '''
    def __init__(self, model, metrics = [], comparisons = [], characteristics = []):
        super(EvaluateDynamicInfo, self).__init__(model, metrics, comparisons, characteristics)
    
    def __eval(self):
        '''
        Computes the bootstrap result
        '''
        
        model = self.model
        
        sample_X, sample_y = self.boot.values()
            
        for (m, mask) in enumerate(self.masks):
            reduced_X_test = (self.X_test.T[mask]).T
            for it in range(self.n_samples):
                model.fit((sample_X[it].T[mask]).T, sample_y[it])
                for (i, metric) in enumerate(self.metrics):
                    #print(i)
                    self.result[i][m][it] = metric.evaluate(model, reduced_X_test, self.y_test)
                
                for (i, comp) in enumerate(self.comparisons):
                    ind = i + len(self.metrics) 
                    #print(i)
                    self.result[ind][m][it] = comp.evaluate(self.full, model, self.X_test,
                                                              reduced_X_test, self.y_test)
                for (i, char) in enumerate(self.characteristics):
                    ind = i + len(self.metrics) + len(self.comparisons)
                    #print(i)
                    self.result[ind][m][it] = char.evaluate(model)

                
                

    def fit(self, X_train, y_train, X_test, y_test, masks = None, n_samples=20, len_sample=None, boot=None):
        '''See EvaluateInfo - fit'''
        super(EvaluateDynamicInfo, self).fit(X_train, y_train, X_test, y_test, masks, n_samples, len_sample)
        if self.len_sample is None:
            self.len_sample = len(X_train)
        if boot is None:
            self.boot = Bootstrap(self.X_train, self.y_train, self.n_samples, self.len_sample)
        else:
            self.boot = boot
        self.__eval()
    

