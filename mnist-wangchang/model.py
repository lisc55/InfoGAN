import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt
from utils import *
import numpy as np

'''
D:
    input 28*28 image
    4*4 conv 64 IRELU stride=2
    4*4 conv 128 IRELU stride=2 BN
    FC 1024 IRELU BN
    FC for D
    FC 128-BN-IRELU_FC output for Q
'''


def get_base(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1.0, 0.02)

    ni = tl.layers.Input(shape)
    nn = tl.layers.Conv2d(64, (4, 4), (2, 2),
                          act=tl.act.lrelu, W_init=w_init, padding='SAME')(ni)
    nn = tl.layers.Conv2d(128, (4, 4), (2, 2),
                          W_init=w_init, padding='SAME')(nn)
    nn = tl.layers.BatchNorm2d(
        decay=0.9, act=tl.act.lrelu, gamma_init=gamma_init)(nn)

    nn = tl.layers.Flatten()(nn)
    nn = tl.layers.Dense(n_units=1024, W_init=w_init)(nn)
    nn = tl.layers.BatchNorm(
        decay=0.9, act=tl.act.lrelu, gamma_init=gamma_init)(nn)
    return tl.models.Model(inputs=ni, outputs=nn, name="first")


def get_discriminator(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1.0, 0.02)
    ndi = tl.layers.Input(shape)
    nqi = tl.layers.Input(shape)

    base = get_base(shape).as_layer()
    base.name = "first"
    nd = base(ndi)
    nq = base(nqi)
    nd = tl.layers.Dense(n_units=1, act=tf.nn.sigmoid,
                         W_init=w_init, name="d1")(nd)

    nq = tl.layers.Dense(n_units=128, W_init=w_init, name="q1")(nq)
    nq = tl.layers.BatchNorm(
        decay=0.9, act=tl.act.lrelu, gamma_init=gamma_init, name="q2")(nq)
    nq = tl.layers.Dense(n_units=14, W_init=w_init, name="q3")(nq)

    return tl.models.Model(inputs=[ndi, nqi], outputs=[nd, nq], name='discriminator')


'''
G:
    Input 74 latent code
    FC 1024 RELU BN
    FC 7*7*128 RELU BN
    4*4 upconv 64 RELU stride 2 BN
    4*4 upconv 1 channel
'''


def get_generator(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1.0, 0.02)
    ni = tl.layers.Input(shape)
    nn = tl.layers.Dense(n_units=1024,
                         W_init=w_init, b_init=None)(ni)
    nn = tl.layers.BatchNorm(decay=0.9, act=tf.nn.relu)(nn)
    nn = tl.layers.Dense(n_units=7 * 7 * 128,
                         W_init=w_init, b_init=None)(nn)
    nn = tl.layers.Reshape([-1, 7, 7, 128])(nn)
    nn = tl.layers.BatchNorm(decay=0.9, act=tf.nn.relu)(nn)

    nn = tl.layers.DeConv2d(
        64, (4, 4), (2, 2), W_init=w_init, b_init=None, padding='SAME')(nn)
    nn = tl.layers.BatchNorm(decay=0.9, act=tf.nn.relu)(nn)
    nn = tl.layers.DeConv2d(
        1, (4, 4), (2, 2), act=tf.nn.tanh, W_init=w_init, b_init=None, padding='SAME')(nn)
    return tl.models.Model(inputs=ni, outputs=nn, name='generator')


class InfoGAN:
    def __init__(self):
        self.D = get_discriminator([None, 28, 28, 1])
        self.G = get_generator([None, 74])

    def cost(self, x, z, c, d):
        EPS = 1e-8
        [d_true, _] = self.D([x, x])
        # label smoothing
        smooth1 = np.random.uniform(0, 0.2, size=d_true.shape)
        fake_image = self.G(tf.concat([z, d, c], 1))
        [d_gen, aux] = self.D([fake_image, fake_image])
        smooth2 = np.random.uniform(0, 0.2, size=d_gen.shape)
        discrete, mean, log_std = aux[:, :10], aux[:, 10:12], aux[:, 12:]
        D_loss = -tf.math.reduce_mean(tf.math.log(d_true + smooth1)) - \
            tf.math.reduce_mean(tf.math.log(1.0 - d_gen + smooth2 + EPS))
        G_loss = tf.math.reduce_mean(-tf.math.log(d_gen + EPS))
        self.g_c_loss = tf.math.reduce_mean(tf.math.reduce_sum(
            log_std + 0.5 * tf.math.square((c - mean) / (tf.math.exp(log_std) + EPS)), axis=1))
        self.g_d_loss = tf.math.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=d, logits=discrete))
        self.mutual_info = self.g_d_loss + self.g_c_loss
        self.g_loss = G_loss
        self.d_loss = D_loss

    def test(self, n):
        z1, c1, d1 = sample(50)
        z2, c2, d2 = sample_d(50)
        z3, c3, d3 = sample_c(50)
        z4, c4, d4 = sample_c2(50)
        fake_image1 = self.G(tf.concat([z1, d1, c1], 1))
        draw(fake_image1, n, 'random')
        fake_image2 = self.G(tf.concat([z2, d2, c2], 1))
        draw(fake_image2, n, 'vary_d')
        fake_image3 = self.G(tf.concat([z3, d3, c3], 1))
        draw(fake_image3, n, 'vary_c1')
        fake_image4 = self.G(tf.concat([z4, d4, c4], 1))
        draw(fake_image4, n, 'vary_c0')
