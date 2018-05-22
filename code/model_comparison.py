
from model_quality import *
from selector import *

class ModelComparison:
    '''Compares several pairs [selection_algorithm x model]'''
    def __init__(self, pairs):
        '''
        Parameters
        ----------
        pairs : array-like, shape = [n_pairs, 2]
                Pairs of (algorithm, model) where algorithm is of class Selector or 
                similar and model has methods fit and predict
        '''
        self.pairs = pairs
        self.model_quality = [ModelQuality(pairs[i][0], pairs[i][1]) for i in range(len(pairs))]
        self.np = len(self.pairs)

    def fit(self, X, y, ns_features=None, test_size=0.3, X_test=None, y_test=None):
        '''
        Fits class to the data.
        
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
        self.ns_features = ns_features
        if X_test is None:
            from sklearn.model_selection import train_test_split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, 
                                                        test_size=test_size, random_state=179)
        else:
            self.X_train = X
            self.y_train = y
            self.X_test = X_test
            self.y_test = y_test
        
        for i in range(self.np):
            self.model_quality[i].fit(self.X_train, self.y_train, self.ns_features, 
                                      X_test=self.X_test, y_test=self.y_test)
    
    def evaluate(self, metrics = [], comparisons = [], characteristics = [], n_samples=20, 
                 len_sample = None, mode='static'):
        '''
        Evaluates the metric values and bootstrap distributions for all pairs (algo, model).
        
        Parameters
        ----------
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
        '''
        
        if mode == 'static':
            all_names = [str(el) for el in metrics] + [str(el) for el in comparisons]
            if len_sample is None:
                len_sample = self.X_test.shape[0]
            self.boot = None
        elif mode == 'dynamic':
            all_names = [str(el) for el in metrics] + [str(el) for el in comparisons] + [str(el) for el in characteristics]
            if len_sample is None:
                len_sample = self.X_train.shape[0]
            self.boot = None
        else:
            raise ValueError("Got unexpected mode %s " % mode)

        self.names = dict(zip(all_names, [i for i in range(len(all_names))]))
        
        for i in range(self.np):
            self.model_quality[i].evaluate(metrics, comparisons, characteristics, n_samples, 
                                                                     len_sample, mode, boot=self.boot)

    def _get_index_by_name(self, name):
        '''Returns the index corresponding to the name of a metric'''
        return self.names[name]
    
    def _draw_one(self, name, params):
        '''Draws one graphic'''
        what, indices = params
        if indices is None:
            indices = np.arange(self.np)
        index = self._get_index_by_name(name)
        
        for i in indices:
            if what == 'quality':
                value = self.model_quality[i].quality[index]
            elif what == 'std':
                value = self.model_quality[i].std[index]
            elif what == 'mean':
                value = self.model_quality[i].mean[index]
            elif what == 'KL':
                value = self.model_quality[i].calculateKL(name)
            elif what == 'rel':
                value = self.model_quality[i].std[index] / self.model_quality[i].mean[index] 
            else:
                raise ValueError("Got unexpected value %s " % what)
            plt.plot(self.ns_features, value,
                     label=str(self.pairs[i][0]) + ';' + str(self.pairs[i][1])[:10])
        
        plt.legend()
        plt.title(name)
        plt.ylabel(what)
    
    def draw(self, names, indices, what):
        '''Draws values for names x metrics
        Parameters
        ----------
        names :  array-like, element type = string
                 Names of metrics to plot
        indices: array-like
                 Indices of pairs (algo, model) to consider
        what :   string
                 'quality', 'mean', 'std', 'KL', 'rel'
        
        '''
        if names is None:
            names = list(self.names.keys())
        plot_graphics(names, self.ns_features, self._draw_one, [what, indices])
    
    def draw_quality(self, names=None, indices=None):
        '''Draws graphics for several metric names'''
        self.draw(names, indices, "quality")
    
    def draw_mean(self, names=None, indices=None):
        '''Draws graphics for several metric names'''
        self.draw(names, indices, "mean")
        
    def draw_std(self, names=None, indices=None):
        '''Draws graphics for several metric names'''
        self.draw(names, indices, "std")
    
    def draw_KL(self, names=None, indices=None):
        '''Draws graphics for several metric names'''
        self.draw(names, indices, "KL")
        
    def draw_relative_std(self, names=None, indices=None):
        self.draw(names, indices, "rel")


# In[ ]:



