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

    def __init__(self, epochs: int = 10000, early_stop_epochs: int = 5, batch_size: int = None,
                 learning_rate: float = 0.00001, velocity: float = 0,
                 decay: float = 0.9):

        # Hyper Parameter
        self.epochs = epochs
        self.early_stop_epochs = early_stop_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.reg = 0.000001
        self.velocity = velocity
        self.decay = decay

        # Tracking
        self.batch_losses_train = []
        self.epoch_losses_train = []
        self.batch_classification_rate_train = []
        self.epoch_classification_rate_train = []

        self.batch_losses_validation = []
        self.epoch_losses_validation = []
        self.batch_classification_rate_validation = []
        self.epoch_classification_rate_validation = []

    @abstractmethod
    def _initialize_weights(self, X: np.array, Y: np.array) -> None:
        return

    @abstractmethod
    def _forward(self, X: np.array, *args, **kwargs) -> np.array:
        """

        Parameters
        ----------
        X

        Returns
        -------

        """

    @abstractmethod
    def _backward(self, X: np.array, T: np.array, Y: np.array, *args, **kwargs) -> None:
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
        if type(probabilities) == tuple:
            probabilities = probabilities[0]
        result = np.zeros_like(probabilities)
        result[np.arange(len(probabilities)), probabilities.argmax(1)] = 1
        return result

    @abstractmethod
    def _backpropagation(self, feature_batch: np.array, target_batch: np.array):
        """

        Parameters
        ----------
        feature_batch
        target_batch

        Returns
        -------

        """

    def fit(self, feature_vector: np.array, target: np.array) -> None:
        """
        Generic fit routine that can be used for a variety of implementations

        Parameters
        ----------
        feature_vector
        target

        Returns
        -------

        """

        if len(target.shape) == 1:
            target = one_hot(target)

        self._initialize_weights(feature_vector, target)
        n, k = feature_vector.shape

        # Split dataset into train and validation
        indices = np.random.permutation(n)
        n_train = int(n * 0.15)
        n_validation = n - n_train
        training_idx, validation_idx = indices[:n_train], indices[n_train:]
        feature_train, feature_validation = feature_vector[training_idx, :], feature_vector[validation_idx, :]
        target_train, target_validation = target[training_idx, :], target[validation_idx, :]
        n_batch_train = n_train // self.batch_size
        n_batch_validation = n_validation // self.batch_size

        # Initialise tracked quantities
        self.batch_losses_train = []
        self.epoch_losses_train = []
        self.batch_classification_rate_train = []
        self.epoch_classification_rate_train = []

        self.batch_losses_validation = []
        self.epoch_losses_validation = []
        self.batch_classification_rate_validation = []
        self.epoch_classification_rate_validation = []

        # Initialise best validation loss. Once this quantity has not been improved for the number of epochs
        # defined as the early_stop_number we break the epoch iteration
        best_validation_loss = np.inf
        epochs_no_improvement = 0

        # Iterate Epochs
        for i in range(self.epochs):

            # --- Iterate training batches for backpropagation ---
            epoch_loss = 0
            epoch_classification = []

            for j in range(n_batch_train):
                feature_batch = feature_train[j * self.batch_size: (j + 1) * self.batch_size, :]
                target_batch = target_train[j * self.batch_size: (j + 1) * self.batch_size, :]

                prediction, batch_loss = self._backpropagation(feature_batch, target_batch)
                epoch_loss += batch_loss
                self.batch_losses_train.append(batch_loss)

                batch_classification_rate = classification_rate(target_batch, prediction)
                epoch_classification.append(batch_classification_rate)
                self.batch_classification_rate_train.append(batch_classification_rate)

            # Epoch analysis
            self.epoch_losses_train.append(epoch_loss)
            self.epoch_classification_rate_train.append(np.mean(epoch_classification))

            # --- Iterate validation batches for scoring ---
            epoch_loss = 0
            epoch_classification = []

            for j in range(n_batch_validation):
                feature_batch = feature_validation[j * self.batch_size: (j + 1) * self.batch_size, :]
                target_batch = target_validation[j * self.batch_size: (j + 1) * self.batch_size, :]

                # Depending on the exact structure of the result of the forward result we may find the desired result
                # as the first item of a tuple
                prediction = self._forward(feature_batch)
                if type(prediction) is tuple:
                    prediction = prediction[0]
                batch_loss = cross_entropy(prediction, target_batch)
                epoch_loss += batch_loss
                self.batch_losses_validation.append(batch_loss)

                batch_classification_rate = classification_rate(target_batch, prediction)
                epoch_classification.append(batch_classification_rate)
                self.batch_classification_rate_validation.append(batch_classification_rate)

            # Epoch analysis
            self.epoch_losses_validation.append(epoch_loss)
            self.epoch_classification_rate_validation.append(np.mean(epoch_classification))

            # We need improvement of at least 0.0001:
            if epoch_loss / best_validation_loss > (1 - 1e-6):
                epochs_no_improvement = 0
                best_validation_loss = epoch_loss
            else:
                epochs_no_improvement += 1

            if epochs_no_improvement >= self.early_stop_epochs:
                print(f'No improvement over {self.early_stop_epochs} epochs: abort training')
                break

    def plot(self):
        # Number of entries for running mean
        n = 100
        ax1 = plt.subplot(3, 2, 1)
        plt.plot(self.epoch_classification_rate_train, label='Train')
        plt.plot(self.epoch_classification_rate_validation, label='Validation')
        plt.legend()
        ax1.title.set_text('Epoch Classification Rate')

        ax2 = plt.subplot(3, 2, 2)
        plt.plot(self.epoch_losses_train, label='Train')
        plt.plot(self.epoch_losses_validation, label='Validation')
        plt.legend()
        ax2.title.set_text('Epoch Losses')

        ax3 = plt.subplot(3, 2, 3)
        y = self.batch_classification_rate_train
        plt.plot(y, label='Train')
        running_mean = np.convolve(y, np.ones((n,)) / n, mode='valid')
        plt.plot(running_mean)
        ax3.title.set_text('Batch Classification Rate Train')

        ax4 = plt.subplot(3, 2, 4)
        y = self.batch_classification_rate_validation
        plt.plot(y, label='Train')
        running_mean = np.convolve(y, np.ones((n,)) / n, mode='valid')
        plt.plot(running_mean)
        ax4.title.set_text('Batch Classification Rate Validation')

        ax5 = plt.subplot(3, 2, 5)
        y = self.batch_losses_train
        plt.plot(y, label='Train')
        running_mean = np.convolve(y, np.ones((n,)) / n, mode='valid')
        plt.plot(running_mean)
        plt.legend()
        ax5.title.set_text('Batch Losses Train')

        ax6 = plt.subplot(3, 2, 6)
        y = self.batch_losses_validation
        plt.plot(y, label='Validation')
        running_mean = np.convolve(y, np.ones((n,)) / n, mode='valid')
        plt.plot(running_mean)
        plt.legend()
        ax6.title.set_text('Batch Losses Validation')

        plt.show()
