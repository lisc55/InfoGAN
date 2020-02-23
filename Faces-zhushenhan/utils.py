import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
import matplotlib
matplotlib.use('Agg')


def sample(size, c1=None, c2=None, c3=None, c4=None, c5=None):
    z = tfd.Uniform(low=-1.0, high=1.0).sample([size, 128])
    if c1 is not None:
        z_con1 = np.array([c1] * size)
        z_con1 = np.reshape(z_con1, [size, 1])
    else:
        z_con1 = tfd.Uniform(low=-1.0, high=1.0).sample([size, 1])
    if c2 is not None:
        z_con2 = np.array([c2] * size)
        z_con2 = np.reshape(z_con2, [size, 1])
    else:
        z_con2 = tfd.Uniform(low=-1.0, high=1.0).sample([size, 1])
    if c3 is not None:
        z_con3 = np.array([c3] * size)
        z_con3 = np.reshape(z_con3, [size, 1])
    else:
        z_con3 = tfd.Uniform(low=-1.0, high=1.0).sample([size, 1])
    if c4 is not None:
        z_con4 = np.array([c4] * size)
        z_con4 = np.reshape(z_con4, [size, 1])
    else:
        z_con4 = tfd.Uniform(low=-1.0, high=1.0).sample([size, 1])
    if c5 is not None:
        z_con5 = np.array([c5] * size)
        z_con5 = np.reshape(z_con5, [size, 1])
    else:
        z_con5 = tfd.Uniform(low=-1.0, high=1.0).sample([size, 1])

    noise = tf.concat([z, z_con1, z_con2, z_con3, z_con4, z_con5], axis=-1)
    return noise, z_con1, z_con2, z_con3, z_con4, z_con5


def train_display_img(model, epoch):
    z1, _, _, _, _, _ = sample(4, -1)
    z2, _, _, _, _, _ = sample(4, -0.5)
    z3, _, _, _, _, _ = sample(4, 0.5)
    z4, _, _, _, _, _ = sample(4, 1)
    z = tf.concat([z1, z2, z3, z4], axis=0)
    model.eval()
    predict = model(z)
    model.train()
    predict = (predict+1.)/2
    plt.figure(figsize=(4, 4))
    plt.suptitle("{}".format(epoch))
    for i in range(predict.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(tf.reshape(predict[i], [32, 32]), cmap="gray")
        plt.axis("off")
    plt.savefig("results/img{:04d}.png".format(epoch))
    plt.close()


def d_loss(real, fake):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(real), real)+cross_entropy(tf.zeros_like(fake), fake)


def g_loss(fake):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake), fake)


def info(fkcon1, z_con1):
    c1 = tf.reduce_mean(tf.reduce_sum(tf.square(fkcon1-z_con1), -1)) * 0.5
    info_loss = c1
    return info_loss
