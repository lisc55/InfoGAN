import numpy as np
import tensorflow as tf
import tensorlayer as tl


def sample(size):
    z = np.random.uniform(-1, 1, size=(size, 62)).astype('float32')
    c = np.random.uniform(-1, 1, size=(size, 2)).astype('float32')
    d = np.zeros((size, 10)).astype('float32')
    idx = np.random.randint(0, 10, size=size)
    d[np.arange(size), idx] = 1
    return z, c, d


def shuffle(x):
    indice = np.random.permutation(len(x))
    return x[indice]


def load_mnist_data():
    X_train, _, _, _, _, _ = tl.files.load_mnist_dataset(
        shape=(-1, 28, 28, 1))
    return X_train
