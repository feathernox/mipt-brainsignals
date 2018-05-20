
# coding: utf-8

# In[1]:

from evaluate_info import *

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def plot_graphics(names, ns_features, func, params=None):
    rows = (len(names)+2) // 2
    plt.figure(figsize=(20, 6 * rows))
    for i in range(len(names)):
        plt.subplot(rows, 2, i+1)
        plt.scatter(ns_features, np.zeros_like(ns_features), label='used n', alpha=0.3, color='red')
        name = names[i]
        if params is None:
            func(name)
        else:
            func(name, params)
        plt.legend()
        plt.title(name)
        plt.xlabel('n')
    plt.show()
        

class ModelQuality():
    '''
    Computes the value and distribution of quality metrics for a pair (feature selection algorthm, model)
    '''
    
    def __init__(self, selector, model):
        '''
        selector - feature selection algorithm. Selector class or similar
        model - desired model
        '''
        
        self.model = model
        self.selector = selector

    def fit(self, X, y, ns_features=None, test_size=0.3, X_test=None, y_test=None):
        '''
        Fits the object to the data
        Parameters
        ----------
        X :           array-like, shape = [n_samples, n_features]
                      Training vectors, where n_samples is the number of samples and
                      n_features is the number of predictors.
        y :           array-like, shape = [n_samples, n_targets]
                      Target vectors, where n_samples is the number of samples and
                      n_targets is the number of response variables.
        ns_features : array-like, 1D. Each element is desired number of features to consider.
                      Default is None.
                      If None, all numbers from 1 to n_features will be considered.
        test_size :   float
                      size of test sample in case test data is generated using test_train_split. 
                      Default is 0.3.
                      If X_test is not None, test_size is ignored
        X_test :      array-like, shape = [n_samples, n_features] or None
                      Test features, where n_samples is the number of samples and
                      n_features is the number of predictors.
                      Default is None.
                      If None, test_train_split is used to generate test sample.
        y_test :      array-like, shape = [n_samples, n_targets] or None
                      Test target vectors, where n_samples is the number of samples and
                      n_targets is the number of response variables.
                      Default is None.
                      If None, test_train_split is used to generate test sample.
        '''

        if X_test is None:
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=179)
        else:
            self.X_train = X
            self.y_train = y
            self.X_test = X_test
            self.y_test = y_test
        
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)
        
        
        if ns_features is None:
            self.ns_features = np.arange(self.X_train.shape[1]) + 1
            #print(ns_features)
        else:
            self.ns_features = np.sort(np.array(ns_features))
        #print(self.ns_features)
        
        #create masks for all ns_features using selection algorithm
        self.masks = np.zeros((len(self.ns_features), self.X_train.shape[1]), dtype=np.bool)
        self.selector.fit(self.X_train, self.y_train)
        for (i, num) in enumerate(self.ns_features):
            cur_features = self.selector.select(num)
            self.masks[i][cur_features] = True 
            
    def _calc_mean_and_std(self):
        '''Calculates the mean and std of result'''
        
        self.mean = np.mean(self.result, axis=-1)
        self.std = np.std(self.result, axis=-1)

    def evaluate(self, metrics = [], comparisons = [], characteristics = [], n_samples=20, 
                 len_sample = None, mode='static', boot=None):
        '''
        Evaluates the metric values and bootstrap distributions.
        
        metrics :         array-like, element type = Metric or similar
                          Desired metrics (see Metric)
        comparisons :     array-like, element type = Comparison or similar
                          Desired comparison (see Comparison)
        characteristics : array-like, element type = Characteristic or similar
                          Desired characteristics (see Characteristic)
        
        n_samples :       int
                          Number of bootstrap samples to generate. Default is 20.
        
        len_sample :      int or None
                          Length of each bootstrap sample
                          If None, it will be equal to the number of features.
        
        mode :            string
                          'static' or 'dynamic'. Default is 'static'
                          The type of EvaluateInfo object. 
                          (See StaticEvaluateInfo/DynamicEvaluateInfo)
        boot:             Bootstrap or None
                          Desired bootstrap sample. If is not None, n_samples and len_sample are ignored.
        
        '''
        
        if mode == 'static':
            self.evaluate_info = EvaluateStaticInfo(self.model, metrics, comparisons)
        else:
            self.evaluate_info = EvaluateDynamicInfo(self.model, metrics, comparisons, characteristics)
        #print(self.evaluate_info)
        self.n_samples = n_samples
        self.len_sample = len_sample
        self.evaluate_info.fit(self.X_train, self.y_train, self.X_test, self.y_test, self.masks, n_samples, len_sample, boot=boot)
        self.result = self.evaluate_info.get_result()
        self.quality = self.evaluate_info.get_quality()
        self._calc_mean_and_std()
    
    def _draw_all(self, names, func):
        '''Draws graphics for several metric names'''
        
        if names is None:
            names = list(self.evaluate_info.names.keys())
        plot_graphics(names, self.ns_features, func)
        
    def _get_index_by_name(self, name):
        '''Returns the index corresponding to the name of a metric'''
        return self.evaluate_info.names[name]
        
    def _draw_one(self, name):
        '''
        Draws values on test sample, mean value on bootstrap samples, std on boostrap samples and values on 
        bootstrap samples for metric with name "name"
        '''
        
        index = self._get_index_by_name(name)
        plt.plot(self.ns_features, self.quality[index], label='quality')
        plt.plot(self.ns_features, self.mean[index], label='mean')
        #plt.plot(self.ns_features, self.mean[index] + self.std[index], color='navy', label='mean+-std')
        plt.fill_between(self.ns_features, self.mean[index] - self.std[index], self.mean[index] + self.std[index],
                         alpha=0.2, label='std', color='blue')
        for i in range(self.n_samples):
            plt.plot(self.ns_features, self.result[index, :, i], color='purple', alpha=0.2)
        plt.ylabel(name)
        
    def draw(self, names=None):
        '''
        Draws values on test sample, mean value on bootstrap samples, std on boostrap samples and values on 
        bootstrap samples for metrics with names in "names"
        If names is None, graphics for all stored metrics will be drawn
        '''
        
        self._draw_all(names, self._draw_one)
    
    def _draw_std_one(self, name):
        '''
        Draws values of std computed on bootstrap samples for metric with name "name"
        '''
        
        index = self._get_index_by_name(name)
        plt.plot(self.ns_features, self.std[index], color='navy')
        plt.ylabel('std')
    
    def draw_std(self, names=None):
        '''
        Draws values of std computed on bootstrap samples for metrics with names in "names"
        If names is None, graphics for all stored metrics will be drawn
        '''
        
        self._draw_all(names, self._draw_std_one)
    
    def check_normality(self, pandas=False, criterion=None, alpha=0.05, return_original_p=False):
        '''
        Checks whether the distribution (computed with bootstrap) of values for all metrics for all numbers
        of features is normal using multiple testing (see multipletest from statsmodels).
        
        Parameters:
        pandas - whether to present the answer as pandas DataFrame. Boolean. Default is False.
        
        criterion - criterion to check normality. Function. If None, normaltest from scipy.stats will be used.
        Default is None.
        
        alpha - importance level used in the criterion. Should be a parameter of criterion. Default is 0.05.
        
        return_original_p - whether to return original p_values (not only corrected)
        
        Return value:
        (result, p, original_p) if return_original_p is True
        (result, p) if return_original_p is False
        
        result - boolean 2D numpy array or pandas DataFrame. True in a cell means that the hypothesis is rejected,
        False - that the hypothesis is not rejected.
        
        p - boolean 2D numpy array or pandas DataFrame consisting of corrected p_values.
        
        original_p - boolean 2D numpy array or pandas DataFrame consisting of original p_values.       
        '''
        
        import scipy.stats as sps
        from statsmodels.sandbox.stats.multicomp import multipletests
        
        if pandas:
            import pandas as pd
        if criterion is None:
            criterion = sps.normaltest
        p_val = np.zeros((len(self.result), self.result.shape[1]))
        for index in range(len(self.result)):
            for ni in range(self.result.shape[1]):
                sample = self.result[index, ni, :]
                p_val[index][ni] = criterion(sample)[1]
        res = multipletests(p_val.ravel(), method='holm', alpha=alpha)
        norm = (res[0]).reshape(p_val.shape).T
        corrected_p = (res[1]).reshape(p_val.shape).T
        p_val = p_val.T
        if pandas:
            norm = pd.DataFrame(norm, index=self.ns_features, columns=self.evaluate_info.names)
            corrected_p = pd.DataFrame(corrected_p, index=self.ns_features, 
                                            columns=self.evaluate_info.names)
            if return_original_p:
                p_val = pd.DataFrame(p_val, index=self.ns_features, columns=self.evaluate_info.names)
        if return_original_p:
            return norm, corrected_p, p_val
        else:
            return norm, corrected_p
    
    def _kullback_leibler(self, a1, s1, a2, s2):
        '''Computes the Kullback-Leibler divergence between two normal distributions'''
        if s1 == 0 or s2 == 0:
            print("Error: std=0")
            return 0
        return np.log(s2/s1) + (s1**2 + (a1 - a2)**2) / (2 * s2**2) - 1/2
    
    def calculateKL(self, name):
        index = self._get_index_by_name(name)
        KL = np.zeros(len(self.ns_features))
        for i in range(1, len(self.ns_features)):
            diff = self._kullback_leibler(self.mean[index][i-1], self.std[index][i-1], 
                                          self.mean[index][i], self.std[index][i])
            KL[i] = KL[i-1] + diff
        return KL
    
    def _draw_KL_one(self, name):
        '''
        Draws sum for metric with name "name", where sum is an accumulated sum of KL divergence between the 
        distribution of values for n_features = i and distribution of values for n_features = i + 1 assuming that both
        distributions are normal (parameters of the distribution are replaced by their estimates).
        '''
        KL = self.calcualteKL(name)
        plt.plot(self.ns_features, KL, label='accumulated KL divergence')
        plt.ylabel('KL')
        
    def draw_KL(self, names=None):
        '''
        Draws sum for metrics with names in "names", where sum is an accumulated sum of KL divergence between the 
        distribution of values for n_features = i and distribution of values for n_features = i + 1 assuming that both
        distributions are normal (parameters of the distribution are replaced by their estimates).
        '''
        self._draw_all(names, self._draw_KL_one)
        
    def summary(self, names=None):
        '''
        Prints summary for names in "names"
        '''
        if names is None:
            names = self.evaluate_info.names
        for name in names:
            index = self._get_index_by_name(name)
            cur_quality = self.quality[index]
            cur_mean = self.mean[index]
            cur_std = self.std[index]
            print(name, ":")
            print("Min metric test value:", cur_quality.min(), "with n =", cur_quality.argmin())
            print("Max metric test value:", cur_quality.max(), "with n =", cur_quality.argmax())
            print("Min metric average value", cur_mean.min(), "with n =", cur_mean.argmin())
            print("Max metric average value", cur_mean.max(), "with n =", cur_mean.argmax())
            print("Min metric std", cur_std.min(), "with n =", cur_std.argmin())
            print("Max metric std", cur_std.max(), "with n =", cur_std.argmax())
            print("Min metric mean-std", (cur_mean-cur_std).min(), "with n =", (cur_mean-cur_std).argmin())
            print("Max metric mean+std", (cur_mean+cur_std).max(), "with n =", (cur_mean+cur_std).argmax())
            print("Min metric relative std", (cur_std/cur_mean).min(), "with n =", (cur_std/cur_mean).argmin())
            print("Max metric relative std", (cur_std/cur_mean).max(), "with n =", (cur_std/cur_mean).argmax())
            print()
            
        
        
        
            


# In[ ]:



