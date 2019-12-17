import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
import matplotlib.pyplot as plt


def sample(size, cat=-1, c1=None, c2=None):
    z = tfd.Uniform(low=-1.0, high=1.0).sample([size, 62])
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
    if cat >= 0:
        z_cat = np.array([cat] * size)
        z_cat = tf.one_hot(z_cat, 10)
    else:
        z_cat = tfd.Categorical(probs=tf.ones([10])*0.1).sample([size, ])
        z_cat = tf.one_hot(z_cat, 10)
    noise = tf.concat([z, z_con1, z_con2, z_cat], axis=-1)
    return noise, z_con1, z_con2, z_cat


def train_display_img(model, epoch):
    z1, _, _, _ = sample(4, 0)
    z2, _, _, _ = sample(4, 1)
    z3, _, _, _ = sample(4, 2)
    z4, _, _, _ = sample(4, 3)
    z = tf.concat([z1, z2, z3, z4], axis=0)
    model.eval()
    predict = model(z)
    model.train()
    predict = (predict+1.)/2
    plt.figure(figsize=(4, 4))
    plt.suptitle("{}".format(epoch))
    for i in range(predict.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(tf.reshape(predict[i], [28, 28]), cmap="gray")
        plt.axis("off")
    plt.savefig("results/img{:04d}.png".format(epoch))
    plt.close()


def d_loss(real, fake):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(real), real)+cross_entropy(tf.zeros_like(fake), fake)


def g_loss(fake):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake), fake)


def info(fkcon1, fkcon2, fkcat, z_con1, z_con2, z_cat):
    c1 = tf.reduce_mean(tf.reduce_sum(tf.square(fkcon1-z_con1), -1)) * 0.5
    c2 = tf.reduce_mean(tf.reduce_sum(tf.square(fkcon2-z_con2), -1)) * 0.5
    sce = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True)(z_cat, fkcat)
    info_loss = c1 + c2 + sce
    return info_loss, c1, c2, sce
