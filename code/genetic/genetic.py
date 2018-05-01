import numpy as np
import time
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import f1_score, r2_score
import matplotlib.pyplot as plt


class GeneticAlgorithm():
    """
    Genetic algorithm for feature selection.
    """

    def __init__(self, mutation_rate = 0.001):
        self.mutation_rate = mutation_rate
        self.kf = KFold(n_splits=5)


    def results(self):
        """
        Debug function.
        Print best results from the fit
        """

        return (generation[0], [idx for idx, gene in enumerate(self.pool[0]) if gene==1])


    def plot_progress(self):
        """
        Debug function.
        Plots the progress of the genetic algorithm
        """

        avs = [np.mean(self.iterations_results[str(x)]['scores']) for x in range(1,101)]
        avs0 = [np.mean(self.iterations_results[str(x)]['scores'][0]) for x in range(1,101)]
        plt.plot(avs, label='Pool Average Score')
        plt.plot(avs0, label='Best Solution Score')
        plt.legend()
        plt.show()


    def _next_generation(generation, mutation_rate):
        ''' Perform a mutation of previous generation.

            Best half of previous generation survives.
            All survived feature sets get mutated with the best one,
            producing two children (feature sets).
        '''
        new_generation = []

        num_survived = (len(generation) + 1) / 2
        all_survived_without_first = generation[1:num_survived]
        survived_first = generation[0]

        for candidate_set in all_survived_without_first:
            split_point = np.random.randint(0, len(candidate_set))
            child_a = np.concatenate((survived_first[:random_split_point], candidate_set[random_split_point:]), axis = 0)
            child_b = np.concatenate((candidate_set[:random_split_point], survived_first[random_split_point:]), axis = 0)

            for idx, gene in enumerate(child_a):
                if np.random.random() < mutation_rate:
                    child_a[idx] = 1 if gene == 0 else 0
            for idx, gene in enumerate(child_b):
                if np.random.random() < mutation_rate:
                    child_b[idx] = 1 if gene == 0 else 0

            new_generation.append(child_a)
            new_generation.append(child_b)

        return new_generation


    def fit(self, model, _type, X, y, num_iterations, generation_size, mutation_rate):
        """
        model = sci-kit learn regression/classification model
        X = X input data
        y = Y output data corresponding to X
        """

        # First generation is chosen randomly.
        # generation[i][k] == whether we added k'th feature to i'th candidate featureset.
        #
        generation = np.random.randint(0, 2, (generation_size, X.shape[1]))

        for iteration in xrange(iterations):
            start_time = time.time()
            scores = []
            fitness = []

            for candidate_set in generation:
                chosen_indices = np.where(candidate_set == 1)
                reduced_X = X[:, chosen_idx]
                if _type == 'regression':
                    score = r2_score(y, model.fit(adj_X,y).predict(adj_X))
                elif _type == 'classification':
                    score = f1_score(y, model.fit(adj_X,y).predict(adj_X))
                scores.append(score)

            fitness = [x / sum(scores) for x in scores]

            fitness, generation, scores = (list(t) for t in zip(*sorted(zip(fitness, [list(l) for l in list(self.pool)], scores),reverse=True)))
            self.iterations_results['{}'.format(iteration)] = dict()
            self.iterations_results['{}'.format(iteration)]['fitness'] = fitness
            self.iterations_results['{}'.format(iteration)]['pool'] = generation
            self.iterations_results['{}'.format(iteration)]['scores'] = scores

            # Perform a mutation
            if iteration + 1 != num_iterations:
                generation = _next_generation(generation, mutation_rate)

            # Debug
            if iteration % 10 == 0:
                duration = round(time.time() - start_time, 2)
                msg = 'Iteration {} completed; Time: {} seconds'
                print(msg.format(iteration, duration))
