import unittest
import matplotlib.pyplot as plt
import numpy as np

from src.utils.activation import softmax, cross_entropy
from src.utils.scoring import classification_rate

np.set_printoptions(linewidth=10000)
import logging
logger = logging.getLogger()
# ===========================================================================


class LogReg:

    def __init__(self, epochs=10000, learning_rate=0.00001, show_plot=False):
        # Weights
        self.W = None
        self.b = None

        # Hyper Parameter
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg = 0.000001
        self.show_plot = show_plot

        self.costs = []
        self.classification_rate = []

    def initialize_weights(self, X: np.array, Y: np.array) -> None:

        N, DX = X.shape
        N_Y, DY = Y.shape
        assert N == N_Y

        self.W = np.random.randn(DX, DY).astype(np.float32) / np.sqrt(DX + DY)
        self.b = np.zeros(DY).astype(np.float32)

    def forward(self, X):
        A1 = X.dot(self.W)
        Y = A1 + self.b
        return softmax(Y)

    def backward(self, X, T, Y):
        self.W += self.learning_rate * (self.derivative_W(X, T, Y) + self.reg * self.W)
        self.b += self.learning_rate * (self.derivative_b(T, Y) + self.reg * self.b)

    @staticmethod
    def derivative_W(X: np.array, T: np.array, Y: np.array) -> np.array:
        eps = T - Y
        return X.T.dot(eps)

    @staticmethod
    def derivative_b(T, Y):
        eps = T - Y
        return eps.sum(axis=0)

    def predict(self, X):
        pY = self.forward(X)
        b = np.zeros_like(pY)
        b[np.arange(len(pY)), pY.argmax(1)] = 1
        return b

    def fit(self, X, T):
        self.initialize_weights(X, T)

        self.costs = []
        self.classification_rate = []

        for i in range(self.epochs):
            Y = self.forward(X)
            self.backward(X, T, Y)
            cost = cross_entropy(T, Y)
            self.costs.append(cost)

            cr = classification_rate(T, Y)
            self.classification_rate.append(cr)

            if (i % 50) == 0:
                logger.debug('Cost: ', cost, ' Classification Rate: ', cr)

        if self.show_plot:
            plt.subplot(2, 1, 1)
            plt.plot(self.costs)
            plt.subplot(2, 1, 2)
            plt.plot(self.classification_rate)
            plt.show()
        return
