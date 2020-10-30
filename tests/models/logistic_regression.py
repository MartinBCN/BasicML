import unittest
import numpy as np

from src.models.logistic_regression import LogReg


class TestStringMethods(unittest.TestCase):

    def setUp(self):
        self.X = np.random.randn(100, 4)
        self.Y = np.random.randn(100, 2)
        self.lr = LogReg()
        self.lr.initialize_weights(self.X, self.Y)

    def test_initialize(self):
        self.assertNotEqual(self.lr.W.shape, np.array(1).shape)
        self.assertNotEqual(self.lr.b.shape, np.array(1).shape)

    def test_forward(self):
        self.Y_hat = self.lr.forward(self.X)
        self.assertEqual(self.Y_hat.shape, self.Y.shape)
        self.assertAlmostEqual(self.Y_hat[0].sum(), 1)

    def test_backward(self):
        # X, T, Y, Z
        T = self.Y
        Y = self.lr.forward(self.X)
        self.lr.backward(self.X, T, Y)
        assert True

    # def test_derivatives(self):
    #     X = self.X
    #     Y = self.lr.forward(self.X)
    #     T = self.Y
    #     # ---------------------------------------------
    #     dW = self.lr.derivative_W(X, T, Y)
    #     self.assertEqual(dW.shape, self.lr.W.shape)
    #     # ---------------------------------------------
    #     db = self.lr.derivative_b(T, Y)
    #     self.assertEqual(db.shape, self.lr.b.shape)


if __name__ == '__main__':
    unittest.main()