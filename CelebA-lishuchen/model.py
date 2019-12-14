import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Input, Dense, Conv2d, DeConv2d, BatchNorm, BatchNorm2d, Reshape, Flatten
from config import flags

def get_G(shape):
	w_init = tf.random_normal_initializer(stddev=0.02)
	gamma_init = tf.random_normal_initializer(1., 0.02)

	ni = Input(shape)
	nn = Dense(n_units=(2 * 2 * 448), W_init=w_init, b_init=None)(ni)
	nn = Reshape(shape=[-1, 2, 2, 448])(nn)
	nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
	nn = DeConv2d(n_filter=256, filter_size=(4, 4), strides=(2, 2), W_init=w_init, b_init=None)(nn)
	nn = BatchNorm2d(decay=0.9, act=tf.nn.relu, gamma_init=gamma_init)(nn)
	nn = DeConv2d(n_filter=128, filter_size=(4, 4), strides=(2, 2), act=tf.nn.relu, W_init=w_init)(nn)
	nn = DeConv2d(n_filter=64, filter_size=(4, 4), strides=(2, 2), act=tf.nn.relu, W_init=w_init)(nn)
	nn = DeConv2d(n_filter=3, filter_size=(4, 4), strides=(2, 2), act=tf.nn.tanh, W_init=w_init)(nn)

	return tl.models.Model(inputs=ni, outputs=nn, name='G')

def get_DQ(shape):
	w_init = tf.random_normal_initializer(stddev=0.02)
	gamma_init = tf.random_normal_initializer(1., 0.02)
	lrelu = lambda x : tf.nn.leaky_relu(x, flags.leaky_rate)
	
	ni = Input(shape)
	nn = Conv2d(n_filter=64, filter_size=(4, 4), strides=(2, 2), act=lrelu, W_init=w_init)(ni)
	nn = Conv2d(n_filter=128, filter_size=(4, 4), strides=(2, 2), W_init=w_init, b_init=None)(nn)
	nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(nn)
	nn = Conv2d(n_filter=256, filter_size=(4, 4), strides=(2, 2), W_init=w_init, b_init=None)(nn)
	nn = BatchNorm2d(decay=0.9, act=lrelu, gamma_init=gamma_init)(nn)
	nn = Flatten()(nn)
	d = Dense(n_units=1, W_init=w_init)(nn)
	q = Dense(n_units=128, W_init=w_init, b_init=None)(nn)
	q = BatchNorm(decay=0.9, act=lrelu, gamma_init=gamma_init)(q)
	q = Dense(n_units=flags.n_categorical * flags.dim_categorical, W_init=w_init)(q)
	
	return tl.models.Model(inputs=ni, outputs=[d, q], name='DQ')
