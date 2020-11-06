from pathlib import Path

from src.classification.classification_nn import ClassificationNN
from src.classification.logistic_regression import LogReg
from src.utils.mnist_reader import load_mnist

import numpy as np
np.set_printoptions(linewidth=1000)


def main():
    data_dir = Path('data/mnist')
    x, y = load_mnist(data_dir, kind='train')
    x = x / 255

    nn = ClassificationNN(epochs=100, batch_size=10, learning_rate=0.0001, early_stop_epochs=10)
    nn.fit(x, y)
    nn.plot()

    lr = LogReg(epochs=100, batch_size=10, learning_rate=0.0001, early_stop_epochs=10)
    lr.fit(x, y)
    lr.plot()


if __name__ == '__main__':
    main()
