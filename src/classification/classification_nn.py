from src.classification.classification_base import BaseClassification

import numpy as np

from src.utils.activation import relu, softmax, cross_entropy


class ClassificationNN(BaseClassification):

    def __init__(self, hidden_layer_size: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hidden_layer_size = hidden_layer_size

    def _initialize_weights(self, X, Y):
        N, DX = X.shape
        N_Y, DY = Y.shape
        assert N == N_Y

        Size_HL = self.hidden_layer_size

        self.W1 = np.random.randn(DX, Size_HL).astype(np.float32) / np.sqrt(DX + Size_HL)
        self.W2 = np.random.randn(Size_HL, DY).astype(np.float32) / np.sqrt(DY + Size_HL)

        self.b1 = np.zeros(Size_HL).astype(np.float32)
        self.b2 = np.zeros(DY).astype(np.float32)

        self.momentum_W1 = self.W1
        self.momentum_W2 = self.W2
        self.momentum_b1 = self.b1
        self.momentum_b2 = self.b2

        # Initialize the cache as ones, this makes switching RMS Prop on/off easier
        self.cache_W1 = np.ones(self.W1.shape)
        self.cache_W2 = np.ones(self.W2.shape)
        self.cache_b1 = np.ones(self.b1.shape)
        self.cache_b2 = np.ones(self.b2.shape)

    def _forward(self, X):
        A1 = X.dot(self.W1)
        Z = A1 + self.b1
        Z = relu(Z)
        A2 = Z.dot(self.W2)
        Y = A2 + self.b2
        return softmax(Y), Z

    def _backward(self, X, T, Y, Z):
        dW1 = self.derivative_W1(X, T, Y, Z)
        db1 = self.derivative_b1(T, Y, Z)
        dW2 = self.derivative_W2(T, Y, Z)
        db2 = self.derivative_b2(T, Y)

        eta = self.learning_rate

        eps = 10 ** -9
        self.momentum_W1 = self.velocity * self.momentum_W1 + (dW1 + self.reg * self.W1) * eta
        self.momentum_b1 = self.velocity * self.momentum_b1 + (db1 + self.reg * self.b1) * eta
        self.momentum_W2 = self.velocity * self.momentum_W2 + (dW2 + self.reg * self.W2) * eta
        self.momentum_b2 = self.velocity * self.momentum_b2 + (db2 + self.reg * self.b2) * eta

        self.W1 += self.momentum_W1
        self.b1 += self.momentum_b1
        self.W2 += self.momentum_W2
        self.b2 += self.momentum_b2

    def _backpropagation(self, feature_batch: np.array, target_batch: np.array):
        prediction, Z = self._forward(feature_batch)
        self._backward(feature_batch, target_batch, prediction, Z)
        batch_loss = cross_entropy(target_batch, prediction)

        return prediction, batch_loss

    def derivative_W1(self, X, T, Y, Z):
        eps = T - Y
        tmp = self.W2.dot(eps.T) * (Z > 0).T
        tmp = tmp.dot(X)
        return tmp.T

    def derivative_b1(self, T, Y, Z):
        eps = T - Y
        tmp = self.W2.dot(eps.T) * (Z > 0).T
        return tmp.T.sum(axis=0)

    def derivative_W2(self, T, Y, Z):
        eps = T - Y
        return Z.T.dot(eps)

    def derivative_b2(self, T, Y):
        eps = T - Y
        return eps.sum(axis=0)