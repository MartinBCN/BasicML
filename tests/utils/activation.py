import unittest
import numpy as np

from src.utils.activation import cross_entropy, softmax


class TestStringMethods(unittest.TestCase):

    def test_cross_entropy(self):
        predictions = np.array([[0.25, 0.25, 0.25, 0.25],
                                [0.01, 0.01, 0.01, 0.96]])
        targets = np.array([[0, 0, 0, 1],
                            [0, 0, 0, 1]])
        answer = 0.71355817782  # Correct answer
        x = cross_entropy(predictions, targets)
        self.assertTrue(np.isclose(x, answer))

    def test_softmax(self):

        # One-dimensional arrays are the easy case
        a = np.array([3.0, 1.0, 0.2])
        result = softmax(a)
        answer = np.array([0.8360188, 0.11314284, 0.05083836])
        self.assertTrue(np.isclose(result, answer).all())

        a = np.array([[3.0, 1.0, 0.2], [3.0, 1.0, 0.2]])
        print(a[0])
        result = softmax(a)
        print(result)
        answer = np.array([[0.8360188, 0.11314284, 0.05083836], [0.8360188, 0.11314284, 0.05083836]])
        self.assertTrue(np.isclose(result, answer).all())


if __name__ == '__main__':
    unittest.main()
