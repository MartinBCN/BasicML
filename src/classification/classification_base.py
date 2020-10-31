from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from src.utils.activation import softmax, cross_entropy
from src.utils.scoring import classification_rate
from src.utils.transform import one_hot

np.set_printoptions(linewidth=10000)
import logging
logger = logging.getLogger()
# ===========================================================================


class BaseClassification(ABC):

    def __init__(self, epochs: int = 10000, batch_size: int = None,
                 learning_rate: float = 0.00001, velocity: float = 0,
                 decay: float = 0.9):

        # Hyper Parameter
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reg = 0.000001
        self.velocity = velocity
        self.decay = decay

        # Tracking
        self.batch_losses = []
        self.epoch_losses = []
        self.batch_classification_rate = []
        self.epoch_classification_rate = []

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
    def _backward(self, X: np.array, T: np.array, Y: np.array) -> None:
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

        if len(target.shape) == 1:
            target = one_hot(target)

        self._initialize_weights(feature_vector, target)
        n, k = feature_vector.shape

        self.batch_losses = []
        self.epoch_losses = []
        self.batch_classification_rate = []
        self.epoch_classification_rate = []

        # Iterate Epochs
        for i in range(self.epochs):

            # Batch Processing
            n_batch = n // self.batch_size
            epoch_loss = 0
            epoch_classification = []

            for j in range(n_batch):
                feature_batch = feature_vector[j * self.batch_size: (j + 1) * self.batch_size, :]
                target_batch = target[j * self.batch_size: (j + 1) * self.batch_size, :]

                prediction = self._forward(feature_batch)
                self._backward(feature_batch, target_batch, prediction)
                batch_loss = cross_entropy(target_batch, prediction)

                epoch_loss += batch_loss
                self.batch_losses.append(batch_loss)

                batch_classification_rate = classification_rate(target_batch, prediction)
                epoch_classification.append(batch_classification_rate)
                self.batch_classification_rate.append(batch_classification_rate)

            # Epoch analysis
            self.epoch_losses.append(epoch_loss)
            self.epoch_classification_rate.append(np.mean(epoch_classification))

    def plot(self):
        plt.subplot(2, 2, 1)
        plt.plot(self.epoch_classification_rate)
        plt.subplot(2, 2, 2)
        plt.plot(self.batch_classification_rate)

        plt.subplot(2, 2, 3)
        plt.plot(self.epoch_losses)
        plt.subplot(2, 2, 4)
        plt.plot(self.batch_losses)
        plt.show()
