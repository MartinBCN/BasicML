from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from src.utils.activation import softmax, cross_entropy
from src.utils.scoring import classification_rate

np.set_printoptions(linewidth=10000)
import logging
logger = logging.getLogger()
# ===========================================================================


class BaseClassification(ABC):

    def __init__(self, epochs=10000, learning_rate=0.00001, show_plot=False):

        # Hyper Parameter
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg = 0.000001

        # Tracking
        self.costs = []
        self.classification_rate = []

    @abstractmethod
    def _initialize_weights(self, X: np.array, Y: np.array) -> None:
        return

    @abstractmethod
    def _forward(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X

        Returns
        -------

        """

    @abstractmethod
    def _backward(self, X: np.array, T: np.array, Y: np.array):
        """

        Parameters
        ----------
        X
        T
        Y

        Returns
        -------

        """

    def predict(self, feature_vector):
        probabilities = self._forward(feature_vector)
        result = np.zeros_like(probabilities)
        result[np.arange(len(probabilities)), probabilities.argmax(1)] = 1
        return result

    def fit(self, feature_vector: np.array, target: np.array) -> None:
        # ToDo: transform target in case it is a vector
        self._initialize_weights(feature_vector, target)

        self.costs = []
        self.classification_rate = []

        for i in range(self.epochs):
            prediction = self._forward(feature_vector)
            self._backward(feature_vector, target, prediction)
            cost = cross_entropy(target, prediction)
            self.costs.append(cost)

            cr = classification_rate(target, prediction)
            self.classification_rate.append(cr)

            if (i % 50) == 0:
                logger.debug('Cost: ', cost, ' Classification Rate: ', cr)

    def plot(self):
        plt.subplot(2, 1, 1)
        plt.plot(self.costs)
        plt.subplot(2, 1, 2)
        plt.plot(self.classification_rate)
        plt.show()
