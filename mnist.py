from pathlib import Path
import numpy as np
np.set_printoptions(linewidth=1000)

from src.classification.logistic_regression import LogReg
from src.utils.mnist_reader import load_mnist
from src.utils.transform import one_hot


def main():
    data_dir = Path('data/mnist')
    x, y = load_mnist(data_dir, kind='train')
    x = x / 255

    y_one_hot = one_hot(y)
    lr = LogReg(epochs=100)
    lr.fit(x, y_one_hot)
    lr.plot()


if __name__ == '__main__':
    main()
