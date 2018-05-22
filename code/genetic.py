import numpy as np
import math
import time
import matplotlib.pyplot as plt


class GeneticAlgorithm():
    """
    Genetic algorithm for feature selection.
    """

    def __init__(self, model, mutation_rate=0.001,
                 num_iterations=1000, generation_size=100):
        self.model = model
        self.mutation_rate = mutation_rate
        self.generation = np.array([])
        self.num_iterations = num_iterations
        self.generation_size = generation_size
        self.iterations_results = []


    def results(self):
        """
        Debug function.
        Print best results from the fit
        """
        return (self.generation[0],
                [idx for idx, gene in enumerate(self.generation[0]) if gene==1])


    def plot_progress(self):
        """
        Debug function.
        Plots the progress of the genetic algorithm
        """
        avs = [np.mean(self.iterations_results[x]['scores']) for x in range(self.num_iterations)]
        avs0 = [np.mean(self.iterations_results[x]['scores'][0]) for x in range(self.num_iterations)]
        plt.plot(avs, label='Pool Average Score')
        plt.plot(avs0, label='Best Solution Score')
        plt.legend()
        plt.show()

    def _gen_initial_generation(self, X):
        self.generation = np.random.randint(
            0, 2, (self.generation_size, X.shape[1]))


    def _gen_next_generation(self):
        '''Perform a mutation of previous generation.
        Best half of previous generation survives.
        All survived feature sets get mutated with the best one,
        producing two children (feature sets).
        '''
        new_generation = []

        num_survived = self.generation_size // 2

        all_survived_without_first = self.generation[1:num_survived]
        survived_first = self.generation[0]
        new_generation.append(survived_first)

        for candidate_set in all_survived_without_first:
            split_point = np.random.randint(0, len(candidate_set))
            child_a = np.concatenate((survived_first[:split_point],
                                      candidate_set[split_point:]), axis=0)
            child_b = np.concatenate((candidate_set[:split_point],
                                      survived_first[split_point:]), axis=0)

            mutated_idx = np.random.binomial(1, self.mutation_rate,
                                             size=child_a.shape)
            child_a[mutated_idx] = 1 - child_a[mutated_idx]

            mutated_idx = np.random.binomial(1, self.mutation_rate,
                                             size=child_b.shape)
            child_b[mutated_idx] = 1 - child_b[mutated_idx]

            new_generation.append(child_a)
            new_generation.append(child_b)

        self.generation = np.array(new_generation)


    def fit(self, X, y):
        """
        model = sci-kit learn regression/classification model
        X = X input data [n_samples x n_features]
        y = Y output data corresponding to X [n_samples x 1]
        """
        X = np.array(X)
        y = np.array(y)
        self.iterations_results = []
        best_scores = []

        # First generation is chosen randomly.
        # generation[i][k] == whether we added k'th feature to i'th candidate featureset.
        self._gen_initial_generation(X)

        for iteration in range(self.num_iterations):
            start_time = time.time()

            scores = []
            for candidate_set in self.generation:
                chosen_indices = np.nonzero(candidate_set)[0]
                reduced_X = X[:, chosen_indices]
                try:
                    self.model.fit(reduced_X, y)
                    current_score = self.model.score(reduced_X, y)
                    scores.append(current_score)

                except:
                    scores.append(0)
                    # print('score - 0')
            scores = np.array(scores)

            assert len(scores) == len(self.generation)

            order = np.argsort(scores)[::-1]
            self.generation = self.generation[order]
            scores = scores[order]

            best_scores.append(scores[0])
            self.iterations_results.append(dict())
            self.iterations_results[iteration]['generation'] = self.generation
            self.iterations_results[iteration]['scores'] = scores

            # Perform a mutation
            if iteration + 1 != self.num_iterations:
                self._gen_next_generation()

            # Debug {{{
            if iteration % 10 == 9:
                duration = round(time.time() - start_time, 2)
                msg = 'Iteration {} completed; Time: {} seconds'
                print(msg.format(iteration, duration))
                if min(best_scores[iteration-9:]) ==\
                        max(best_scores[iteration-9:]):
                    break

            # }}}

        return self

    def _best_candidate_set(self, max_num_features=-1):
        best_candidate_set = None
        best_score = 0

        for iteration in range(len(self.iterations_results)):
            results = self.iterations_results[iteration]
            for i, candidate_set in enumerate(results['generation']):
                score = results['scores'][i]
                chosen_indices = np.nonzero(candidate_set)[0]
                size = len(chosen_indices)

                if size <= max_num_features or max_num_features == -1:
                    if score > best_score:
                        best_candidate_set = candidate_set
                        best_score = score
        return best_candidate_set, best_score

    def select(self, max_num_features):
        indices = self._best_candidate_set(max_num_features)[0]
        return indices

    def transform(self, X, max_num_features = -1):
        indices = self._best_candidate_set(max_num_features)
        return np.array([X[i] for i in indices])

    def fit_transform(self, X, y, max_num_features = -1):
        self.fit(X, y)
        return self.transform(X, max_num_features)

