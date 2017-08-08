"""
======================== cProfile Results of Interest ========================
============================ Generated 01/31/2017 ============================

31,522,436 function calls (31,518,492 primitive calls) in 100.702 seconds

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    
  1487994    3.788    0.000   32.834    0.000 fuzzy_art.py:137(fuzzy_AND)
  2947472    3.133    0.000   52.574    0.000 fuzzy_art.py:142(norm)
    14258    0.161    0.000    0.460    0.000 fuzzy_art.py:148(augment_compliments)
  1366311    8.387    0.000   87.156    0.000 fuzzy_art.py:154(category_choice)
   107425    0.302    0.000    6.622    0.000 fuzzy_art.py:159(passes_vigilance_test)
    14258    0.093    0.000    0.412    0.000 fuzzy_art.py:163(update_weights)
    
==============================================================================
"""

import numpy as np
from copy import deepcopy
from time import time


# TODO - implement pruning
class FuzzyConfig:
    
    def __init__(self, params: dict=None):
        """
        Input parameter 'params', if specified, should be of the following form:
            
        "data"      => REQUIRED, np.array   : training data, np.ndarray with values between 0 and 1
        "rho"       => REQUIRED, float      : vigilance parameter rho between 0 and 1
        "alpha"     => REQUIRED, float      : signal rule parameter greater than 0
        "beta"      => OPTIONAL, float      : learning rate parameter, defaulted to 1 for fast learning
        "epochs"    => OPTIONAL, int        : maximum number of training epochs, default 100 (should be plenty)
        "prune"     => OPTIONAL, bool       : whether inactive nodes should be pruned after training
        "weights"   => OPTIONAL, np.array   : weights to use in training if provided, default is all ones
        "clusters"  => OPTIONAL, int        : number of previously found clusters
        """

        # TODO - look into making pack/unpack methods to save all the repetitive assignments

        # use initialization values if provided
        if params is not None:
            self.data = params["data"]
            self.rho = params["rho"]
            self.alpha = params["alpha"]
            self.beta = params["beta"] if "beta" in params else 1.0
            self.epochs = params["epochs"] if "epochs" in params else 100
            self.prune = params["prune"] if "prune" in params else False
            self.num_clusters = params["clusters"] if "clusters" in params else 0
            self.weights = params["weights"] if "weights" in params else np.ones((1, self.data.shape[1] * 2))
            self.verbose = params["verbose"] if "verbose" in params else False
            
        # if initialization values not provided, used default values
        else:
            self.data = None
            self.rho = None
            self.alpha = None
            self.beta = 1.0
            self.epochs = 100
            self.prune = False
            self.weights = np.ones((1, self.data.shape[1] * 2))
            self.num_clusters = 0
            self.verbose = False

    def __str__(self):
        return str("Alpha: %f\tBeta: %f\tRho: %f\tEpochs: %d" % (self.alpha, self.beta, self.rho, self.epochs))


class FuzzyART:

    def __init__(self, config: FuzzyConfig):
        self.config = config

    def train(self, should_print=False) -> tuple:
        data = self.config.data

        assert np.min(data) >= 0 and np.max(data) <= 1, "Data must be normalized between 0 and 1 to use FuzzyART."

        weights = np.vstack((self.config.weights, np.ones((1, self.config.weights.shape[1]))))#self.config.weights
        alpha = self.config.alpha
        beta = self.config.beta
        rho = self.config.rho
        epochs = self.config.epochs
        num_clusters = self.config.num_clusters
        (num_genes, num_features) = data.shape if len(data.shape) == 2 else (1, data.shape)
        if should_print: print("[STATUS] Running FuzzyART with {0} genes and {1} samples.".format(num_genes, num_features))

        # store a backup of the old weights
        old_weights = deepcopy(weights)

        found_clusters = np.zeros((num_genes), np.int32)
        current_iteration = 1

        start_time = time()

        while current_iteration <= epochs:

            for gene in range(num_genes):
                # augment complimentary coding on current row of data
                input_array = FuzzyART.augment_compliments(data[gene, :])

                # allocate memory to layer F2, leave empty for speed since values will be overwritten immediately
                layer_F2 = np.empty((num_clusters + 1))

                # calculate category choice function for layer 2 of fuzzy ART
                for cluster in range(layer_F2.shape[0]):
                    layer_F2[cluster] = FuzzyART.category_choice(input_array, weights[cluster, :], alpha)

                while True:
                    winningNode = np.argmax(layer_F2)

                    # perform vigilance test
                    if FuzzyART.passes_vigilance_test(input_array, weights[winningNode, :], rho):
                        found_clusters[gene] = int(winningNode)
                        break
                    layer_F2[winningNode] = 0

                # update the weights based on new information
                # here lies the source of many hours of frustration, all because I typed rho instead of beta by mistake.
                weights[winningNode, :] = FuzzyART.update_weights(input_array, weights[winningNode, :], beta)

                # if winningNode == config.num_clusters:
                if winningNode == layer_F2.shape[0] - 1:
                    num_clusters += 1
                    weights = np.vstack((weights, np.ones((1, weights.shape[1]))))

            # stop if weights have not changed
            if weights.shape == old_weights.shape:
                if np.all(np.isclose(weights, old_weights, atol=1e-8)):
                    if should_print: print("[STATUS] Training has converged after {0} iterations".format(current_iteration))
                    # slice last row off weights, since last row corresponds to uncommitted node
                    return weights[:-1, :], found_clusters, num_clusters

            old_weights = deepcopy(weights)
            if should_print: print("[STATUS] Epoch {0} finished.\tElapsed time: {1} seconds.".format(
                current_iteration, round(time() - start_time, 2)))
            current_iteration += 1

        if should_print: print("[STATUS] Maximum number of epochs reached and training has stopped.")

        # slice last row off weights, since last row corresponds to uncommitted node
        np.savetxt("weights.csv", weights[:-1, :])
        np.savetxt("clusters.csv", found_clusters)
        return weights[:-1, :], found_clusters, num_clusters


    @staticmethod
    def fuzzy_AND(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.min(np.asarray([a, b]), axis=0)

    @staticmethod
    def norm(a: np.ndarray, order: int=1) -> float:
        # ~26.2 microseconds per loop on random array of size (1,144), as of 01/26/2017
        return np.linalg.norm(a, ord=order)

    # doubles the size of each row, appending each row element's compliment directly after it
    @staticmethod
    def augment_compliments(a: np.ndarray) -> np.ndarray:
        a_comp = np.subtract(np.ones(a.shape), a)
        return np.asarray([a, a_comp]).ravel('F')

    # input_array and weights must both be 1d arrays, alpha between 0 and 1.
    # TODO - this is taking over 50% of execution time, look into speedups or memoization.
    @staticmethod
    def category_choice(input_array: np.ndarray, weights: np.ndarray, alpha: float) -> float:
        return np.divide(FuzzyART.norm(FuzzyART.fuzzy_AND(input_array, weights)), (alpha + FuzzyART.norm(weights)))

    # returns true if the row passes a vigilance test with given vigilance parameter rho
    @staticmethod
    def passes_vigilance_test(input_array: np.ndarray, row_weights: np.ndarray, rho: float) -> bool:
        return FuzzyART.norm(FuzzyART.fuzzy_AND(input_array, row_weights)) >= rho * FuzzyART.norm(input_array)

    @staticmethod
    def update_weights(input_array: np.ndarray, row_weights: np.ndarray, beta: float) -> np.ndarray:
        # take a shortcut if beta is 1 (if we are using fast learning)
        if beta == 1:
            return beta * FuzzyART.fuzzy_AND(input_array, row_weights)
        # otherwise, do the whole calculation
        else:
            return beta * FuzzyART.fuzzy_AND(input_array, row_weights) + (1 - beta) * row_weights
