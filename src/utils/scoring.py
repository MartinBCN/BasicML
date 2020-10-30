import numpy as np


def classification_rate(y_true: np.array, y_pred: np.array):
    """
    Classification Rate for one-hot encoded arrays

    Parameters
    ----------
    y_true
    y_pred

    Returns
    -------

    """
    v_true = np.argmax(y_true, axis=1)
    v_pred = np.argmax(y_pred, axis=1)
    return np.mean(v_true == v_pred)
