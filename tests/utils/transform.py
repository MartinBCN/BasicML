import unittest
import numpy as np

from src.utils.activation import cross_entropy, softmax
from src.utils.transform import one_hot


class TestStringMethods(unittest.TestCase):

    def test_one_hot(self):
        x = np.array([1, 2, 3])
        answer = np.array([[0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        result = one_hot(x)

        self.assertTrue(np.isclose(result, answer).all())

        # Use a fifth class that does not appear in the sample:
        answer = np.array([[0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0]])
        result = one_hot(x, number_classes=5)

        self.assertTrue(np.isclose(result, answer).all())


if __name__ == '__main__':
    unittest.main()
