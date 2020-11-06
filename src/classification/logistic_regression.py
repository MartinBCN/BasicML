import matplotlib.pyplot as plt
import numpy as np

from src.classification.classification_base import BaseClassification
from src.utils.activation import softmax, cross_entropy

np.set_printoptions(linewidth=10000)
import logging
logger = logging.getLogger()
# ===========================================================================


class LogReg(BaseClassification):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Weights
        self.W = None
        self.b = None

    def _initialize_weights(self, feature_vector: np.array, target: np.array) -> None:

        n, k = feature_vector.shape
        n_target, classes = target.shape
        assert n == n_target

        self.W = np.random.randn(k, classes).astype(np.float32) / np.sqrt(k + classes)
        self.b = np.zeros(classes).astype(np.float32)

    def _forward(self, X, *args, **kwargs):
        A1 = X.dot(self.W)
        Y = A1 + self.b
        return softmax(Y)

    def _backward(self, X, T, Y, *args, **kwargs):
        self.W += self.learning_rate * (self._derivative_W(X, T, Y) + self.reg * self.W)
        self.b += self.learning_rate * (self._derivative_b(T, Y) + self.reg * self.b)

    def step(self, feature_batch: np.array, target_batch: np.array):
        prediction = self._forward(feature_batch)
        self._backward(feature_batch, target_batch, prediction)
        batch_loss = cross_entropy(target_batch, prediction)

        return prediction, batch_loss

    @staticmethod
    def _derivative_W(X: np.array, T: np.array, Y: np.array) -> np.array:
        eps = T - Y
        return X.T.dot(eps)

    @staticmethod
    def _derivative_b(T, Y):
        eps = T - Y
        return eps.sum(axis=0)
