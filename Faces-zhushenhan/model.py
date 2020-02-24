import tensorflow as tf
import tensorlayer as tl
from config import flags


def Generator(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1.0, 0.02)
    ni = tl.layers.Input(shape)
    nn = tl.layers.Dense(n_units=1024, b_init=None, W_init=w_init)(ni)
    nn = tl.layers.BatchNorm(decay=0.9, act=tf.nn.relu,
                             gamma_init=gamma_init)(nn)
    nn = tl.layers.Dense(n_units=8*8*128, b_init=None, W_init=w_init)(nn)
    nn = tl.layers.BatchNorm(decay=0.9, act=tf.nn.relu,
                             gamma_init=gamma_init)(nn)
    nn = tl.layers.Reshape([-1, 8, 8, 128])(nn)
    nn = tl.layers.DeConv2d(64, (4, 4), strides=(
        2, 2), padding="SAME", W_init=w_init)(nn)
    nn = tl.layers.BatchNorm(decay=0.9, act=tf.nn.relu,
                             gamma_init=gamma_init)(nn)
    nn = tl.layers.DeConv2d(
        1, (4, 4), strides=(2, 2), padding="SAME", act=tf.nn.sigmoid, W_init=w_init)(nn)
    return tl.models.Model(inputs=ni, outputs=nn)


def Discriminator(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1.0, 0.02)
    ni = tl.layers.Input(shape)
    nn = tl.layers.Conv2d(64, (4, 4), strides=(
        2, 2), act=lambda x: tl.act.lrelu(x, flags.leaky_rate), padding="SAME", W_init=w_init)(ni)
    nn = tl.layers.Conv2d(128, (4, 4), strides=(
        2, 2), padding="SAME", W_init=w_init)(nn)
    nn = tl.layers.BatchNorm2d(decay=0.9, act=lambda x: tl.act.lrelu(
        x, flags.leaky_rate), gamma_init=gamma_init)(nn)
    nn = tl.layers.Flatten()(nn)
    nn = tl.layers.Dense(n_units=1024, W_init=w_init)(nn)
    nn = tl.layers.BatchNorm(decay=0.9, act=lambda x: tl.act.lrelu(
        x, flags.leaky_rate), gamma_init=gamma_init)(nn)

    mid = nn
    nn = tl.layers.Dense(n_units=1, W_init=w_init)(nn)
    return tl.models.Model(inputs=ni, outputs=[nn, mid])


def q_sample(mu, var):
    unit = tf.random.normal(shape=mu.shape)
    sigma = tf.sqrt(var)
    return mu+sigma*unit


def Auxiliary(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1.0, 0.02)
    ni = tl.layers.Input(shape)
    con1_mu = tl.layers.Dense(n_units=2, W_init=w_init)(ni)
    con1_var = tl.layers.Dense(n_units=2, W_init=w_init)(ni)
    return tl.models.Model(inputs=ni, outputs=[con1_mu, con1_var])
