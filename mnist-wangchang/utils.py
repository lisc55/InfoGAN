import numpy as np
import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt


def sample(size):
    z = np.random.normal(size=(size, 62)).astype('float32')
    c = np.random.normal(size=(size, 2)).astype('float32')
    d = np.zeros((size, 10)).astype('float32')
    idx = np.random.randint(0, 10, size=size)
    d[np.arange(size), idx] = 1
    return z, c, d


def sample_d(size):
    z = np.random.normal(size=(1, 62)).astype('float32')
    z = np.tile(z, size).reshape((size, 62))
    c = np.random.normal(size=(1, 2)).astype('float32')
    c = np.tile(c, size).reshape((size, 2))
    d = np.zeros((size, 10)).astype('float32')
    for i in range(size):
        d[i][i % 10] = 1
    return z, c, d


def sample_c(size):
    z = np.random.normal(size=(1, 62)).astype('float32')
    z = np.tile(z, size).reshape((size, 62))
    c = np.random.normal(size=(1, 2)).astype('float32')
    c = np.tile(c, size).reshape((size, 2))
    d = np.zeros((size, 10)).astype('float32')
    for i in range(size):
        d[i][0] = 1
    for i in range(size):
        if i >= size // 2:
            c[i][1] = 0.8 + 0.08 * (i - size // 2)
        else:
            c[i][1] = -0.8 - 0.08 * i
    return z, c, d


def sample_c2(size):
    z = np.random.normal(size=(1, 62)).astype('float32')
    z = np.tile(z, size).reshape((size, 62))
    c = np.random.normal(size=(1, 2)).astype('float32')
    c = np.tile(c, size).reshape((size, 2))
    d = np.zeros((size, 10)).astype('float32')
    for i in range(size):
        d[i][0] = 1
    for i in range(size):
        if i >= size // 2:
            c[i][0] = 0.8 + 0.08 * (i - size // 2)
        else:
            c[i][0] = -0.8 - 0.08 * i
    return z, c, d


def load_mnist_data():
    X_train, _, _, _, _, _ = tl.files.load_mnist_dataset(
        shape=(-1, 28, 28, 1))
    return X_train


def draw(fake_image, n, name):
    plt.figure()
    for i in range(1, 50 + 1):
        plt.subplot(5, 10, i)
        t = np.reshape(fake_image[i - 1], (28, 28))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(t, cmap='gray')
    plt.savefig('results/' + name + str(n) + '.jpg')
    plt.clf()
    plt.close()
