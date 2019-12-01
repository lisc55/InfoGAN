import tensorflow as tf
import tensorlayer as tl

'''
D:
    input 28*28 image
    4*4 conv 64 IRELU stride=2
    4*4 conv 128 IRELU stride=2 BN
    FC 1024 IRELU BN
    FC for D
    FC 128-BN-IRELU_FC output for Q
'''


def get_discriminator(shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1.0, 0.02)

    ni = tl.layers.Input(shape)
    nn = tl.layers.Conv2d(64, (4, 4), (2, 2),
                          act=tl.act.lrelu, W_init=w_init)(ni)
    nn = tl.layers.Conv2d(128, (4, 4), (2, 2), W_init=w_init)(nn)
    nn = tl.layers.BatchNorm2d(
        decay=0.9, act=tl.act.lrelu, gamma_init=gamma_init)(nn)

    nn = tl.layers.Flatten()(nn)
    nn = tl.layers.Dense(n_units=1024, W_init=w_init)(nn)
    nn = tl.layers.BatchNorm(
        decay=0.9, act=tl.act.lrelu, gamma_init=gamma_init)(nn)
    nd = tl.layers.Dense(n_units=1, act=tf.nn.sigmoid, W_init=w_init)(nn)

    nq = tl.layers.Dense(n_units=128, W_init=w_init)(nn)
    nq = tl.layers.BatchNorm(
        decay=0.9, act=tl.act.lrelu, gamma_init=gamma_init)(nq)
    nq = tl.layers.Dense(n_units=14, W_init=w_init)(nq)

    D = tl.models.Model(inputs=ni, outputs=nd, name='discriminator')
    Q = tl.models.Model(inputs=ni, outputs=nq, name='recognitor')
    return D, Q


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
    nn = tl.layers.BatchNorm(decay=0.9, act=tl.act.lrelu)(nn)
    nn = tl.layers.Dense(n_units=7*7*128,
                         W_init=w_init, b_init=None)(nn)
    nn = tl.layers.Reshape([-1, 7, 7, 128])(nn)
    nn = tl.layers.BatchNorm(decay=0.9, act=tl.act.lrelu)(nn)

    nn = tl.layers.DeConv2d(64, (4, 4), (2, 2), W_init=w_init, b_init=None)(nn)
    nn = tl.layers.BatchNorm(decay=0.9, act=tl.act.lrelu)(nn)
    nn = tl.layers.DeConv2d(
        1, (4, 4), (2, 2), act=tf.nn.tanh, W_init=w_init, b_init=None)(nn)
    return tl.models.Model(inputs=ni, outputs=nn, name='generator')


class InfoGAN:
    def __init__(self):
        self.D, self.Q = get_discriminator([None, 28, 28, 1])
        self.G = get_generator([None, 74])

    def cost(self, x, z, c, d):
        d_true = self.D(x)
        fake_image = self.G(tf.concat([z, d, c], 1))
        d_gen = self.D(fake_image)
        aux = self.Q(fake_image)
        discrete, mean, log_std = aux[:, :10], aux[:, 10:12], aux[:, 12:]
        D_loss = -tf.math.reduce_mean(tf.math.log(d_true)) - \
            tf.math.reduce_mean(tf.math.log(1.0-d_gen))
        G_loss = tf.math.reduce_mean(-tf.math.log(d_gen))
        self.g_c_loss = tf.math.reduce_mean(tf.math.reduce_sum(
            log_std+0.5*tf.math.square((c-mean)/tf.math.exp(log_std)), axis=1))
        self.g_d_loss = tf.math.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=d, logits=discrete))
        self.mutual_info = self.g_d_loss+self.g_c_loss
        self.g_loss = G_loss
        self.d_loss = D_loss
