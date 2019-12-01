from utils import *
from model import *
import tensorflow as tf
import tensorlayer as tl
import os
import time
import multiprocessing
import numpy as np

NEPOCH = 100
train_data = load_mnist_data()
train_data = np.reshape(train_data, (-1, 50, 28, 28, 1))
model = InfoGAN()
model.G.train()
model.D.train()
model.Q.train()
d_optimizer = tf.optimizers.Adam(2e-4, 0.5)
g_optimizer = tf.optimizers.Adam(1e-3, 0.5)
q_optimizer = tf.optimizers.Adam(2e-4, 0.5)
for epoch in range(NEPOCH):
    for step, batch_images in enumerate(train_data):
        if batch_images.shape[0] != 50:
            break
        step_time = time.time()
        with tf.GradientTape(persistent=True) as tape:
            z, c, d = sample(50)
            model.cost(batch_images, z, c, d)
            d_loss, g_loss, mutual_info = model.d_loss, model.g_loss, model.mutual_info
        grad = tape.gradient(g_loss, model.G.trainable_weights)
        g_optimizer.apply_gradients(zip(grad, model.G.trainable_weights))
        grad = tape.gradient(d_loss, model.D.trainable_weights)
        d_optimizer.apply_gradients(zip(grad, model.D.trainable_weights))
        grad = tape.gradient(mutual_info, model.Q.trainable_weights)
        d_optimizer.apply_gradients(zip(grad, model.Q.trainable_weights))
        del tape

        print("Epoch: [{}/{}] [{}] took: {:.3f}, d_loss: {:.5f}, g_loss: {:.5f}".format(epoch,
                                                                                        NEPOCH, step, time.time()-step_time, d_loss, g_loss))
