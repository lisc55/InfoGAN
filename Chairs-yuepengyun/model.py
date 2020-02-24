import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from data import flags
from tensorlayer.layers import Input, Dense, DeConv2d, Reshape, BatchNorm2d, Conv2d, Flatten, BatchNorm


def get_G(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    ni = Input(shape)
    nn = Dense(n_units=(1024), W_init=w_init, b_init=None)(ni)
    nn = Reshape(shape=[-1, 1024])(nn)
    nn = BatchNorm(decay=0.9, act=tf.nn.relu,
                     gamma_init=gamma_init, name=None)(nn)
    nn = Dense(n_units=(8 * 8 * 256), W_init=w_init, b_init=None)(nn)
    nn = Reshape(shape=[-1, 8, 8, 256])(nn)  # ???
    nn = BatchNorm(decay=0.9, act=tf.nn.relu,
                     gamma_init=gamma_init, name=None)(nn)
    nn = DeConv2d(n_filter=256, filter_size=(4, 4),
                  strides=(1, 1), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
    nn = DeConv2d(n_filter=256, filter_size=(4, 4),
                  strides=(2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
    nn = DeConv2d(n_filter=128, filter_size=(4, 4),
                  strides=(2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
    nn = DeConv2d(n_filter=64, filter_size=(4, 4), strides=(
        2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
    nn = DeConv2d(n_filter=1, filter_size=(4, 4), strides=(
        1, 1), act=tf.nn.sigmoid, W_init=w_init)(nn)

    return tl.models.Model(inputs=ni, outputs=nn, name='G')


def get_D(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    def lrelu(x): return tf.nn.leaky_relu(x, flags.leaky_rate)

    ni = Input(shape)
    nn = Conv2d(n_filter=64, filter_size=(4, 4),
                strides=(2, 2), act=lrelu, W_init=w_init)(ni)
    nn = Conv2d(n_filter=128, filter_size=(4, 4), strides=(
        2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(nn)
    nn = Conv2d(n_filter=256, filter_size=(4, 4), strides=(
        2, 2), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(nn)
    nn = Conv2d(n_filter=256, filter_size=(4, 4), strides=(
        1, 1), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(nn)
    nn = Conv2d(n_filter=256, filter_size=(4, 4), strides=(
        1, 1), W_init=w_init, b_init=None)(nn)
    nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(nn)
    nn = Flatten()(nn)
    nn = Dense(n_units=1024, W_init=w_init)(nn)
    nn = BatchNorm(decay=0.9, act=lrelu, gamma_init=gamma_init)(nn)
    mid = nn
    d = Dense(n_units=1, W_init=w_init)(nn)
    return tl.models.Model(inputs=ni, outputs=[d, mid], name='D')

def get_Q(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)

    ni = Input(shape)
    cat1 = Dense(n_units=flags.dim_categorical, W_init=w_init)(ni)
    cat2 = Dense(n_units=flags.dim_categorical, W_init=w_init)(ni)
    cat3 = Dense(n_units=flags.dim_categorical, W_init=w_init)(ni)
    mu = Dense(n_units=1, W_init=w_init, name='mu')(ni)
    # sigma = Dense(n_units=1, W_init=w_init, name='sigma')(ni)
    return tl.models.Model(inputs=ni, outputs=[cat1, cat2, cat3, mu], name='Q')


# def get_D(shape):
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     # gamma_init = tf.random_normal_initializer(1., 0.02)
#     # def lrelu(x): return tf.nn.leaky_relu(x, flags.leaky_rate)

#     base_layer = get_base(shape).as_layer()
#     ni = Input(shape)
#     d = base_layer(ni)
#     d = Dense(n_units=1, W_init=w_init)(d)

#     return tl.models.Model(inputs=ni, outputs=d, name='D')


# def get_Q(shape):
#     w_init = tf.random_normal_initializer(stddev=0.02)
#     # gamma_init = tf.random_normal_initializer(1., 0.02)
#     # def lrelu(x): return tf.nn.leaky_relu(x, flags.leaky_rate)

#     base_layer = get_base(shape).as_layer()
#     ni = Input(shape)
#     q = base_layer(ni)
#     cat1 = Dense(n_units=flags.dim_categorical, W_init=w_init)(q)
#     cat2 = Dense(n_units=flags.dim_categorical, W_init=w_init)(q)
#     cat3 = Dense(n_units=flags.dim_categorical, W_init=w_init)(q)
#     mu = Dense(n_units=1, W_init=w_init)(q)
#     var = Dense(n_units=1, W_init=w_init)(q)
#     return tl.models.Model(inputs=ni, outputs=[cat1, cat2, cat3, mu, var], name='Q')
