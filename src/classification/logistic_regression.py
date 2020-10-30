import matplotlib.pyplot as plt
import numpy as np

from src.classification.classification_base import BaseClassification
from src.utils.activation import softmax, cross_entropy
from src.utils.scoring import classification_rate

np.set_printoptions(linewidth=10000)
import logging
logger = logging.getLogger()
# ===========================================================================


class LogReg(BaseClassification):

    def __init__(self, epochs=10000, learning_rate=0.00001):
        super(BaseClassification, self).__init__()
        # Weights
        self.W = None
        self.b = None

        # Hyper Parameter
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reg = 0.000001

    def _initialize_weights(self, X: np.array, Y: np.array) -> None:

        N, DX = X.shape
        N_Y, DY = Y.shape
        assert N == N_Y

        self.W = np.random.randn(DX, DY).astype(np.float32) / np.sqrt(DX + DY)
        self.b = np.zeros(DY).astype(np.float32)

    def _forward(self, X):
        A1 = X.dot(self.W)
        Y = A1 + self.b
        return softmax(Y)

    def _backward(self, X, T, Y):
        self.W += self.learning_rate * (self._derivative_W(X, T, Y) + self.reg * self.W)
        self.b += self.learning_rate * (self._derivative_b(T, Y) + self.reg * self.b)

    @staticmethod
    def _derivative_W(X: np.array, T: np.array, Y: np.array) -> np.array:
        eps = T - Y
        return X.T.dot(eps)

    @staticmethod
    def _derivative_b(T, Y):
        eps = T - Y
        return eps.sum(axis=0)
