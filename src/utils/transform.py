import numpy as np


def one_hot(y: np.array, number_classes: int = None) -> np.array:
    n = y.shape[0]

    if number_classes is None:
        classes = np.unique(y)
        number_classes = np.max(classes) + 1

    y_one_hot = np.zeros((n, number_classes))
    y_one_hot[np.arange(n), y] = 1
    return y_one_hot
