import numpy as np
np.set_printoptions(linewidth=1000)


def softmax(vector: np.array) -> np.array:
    """
    My own implementation of the softmax function

    Parameters
    ----------
    vector

    Returns
    -------

    """

    # Notice that
    # exp(x_i)/sum_i exp(x_i)
    # = [exp(-m)/exp(-m)]exp(x_i)/sum_i exp(x_i)
    # = exp(x_i - m)/sum_i exp(x_i-m)

    # I add some gymnastics to allow this to work on vectors as well as arrays
    if len(vector.shape) == 1:
        vector = vector.reshape(1, -1)
        reshape = True
    else:
        reshape = False

    nominator = np.exp(vector - np.max(vector))
    denominator = nominator.sum(axis=1, keepdims=True)

    result = nominator / denominator

    if reshape:
        result = result.reshape(-1,)
    return result


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def cross_entropy(predictions, targets, epsilon=1e-12):
    """

    Parameters
    ----------
    predictions: np.array(N, k)
    targets: np.array(N, k)
    epsilon: float
        Correction factor for numerical stability

    Returns
    -------

    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    result = -np.sum(targets*np.log(predictions+1e-9))/N
    return result


def relu(X):
    # np.maximum keeps the shape of X! np.max returns less dimensional objects
    return np.maximum(X, 0)
