from utils import *
from model import *
import tensorflow as tf
import tensorlayer as tl
import os
import time
import multiprocessing
import numpy as np

NEPOCH = 100
batch_size = 50
train_data = load_mnist_data()
train_data = np.reshape(train_data, (-1, batch_size, 28, 28, 1))
model = InfoGAN()
model.G.train()
model.D.train()
d_optimizer = tf.optimizers.Adam(2e-4, 0.5)
g_optimizer = tf.optimizers.Adam(2e-4, 0.5)
q_optimizer = tf.optimizers.Adam(2e-4, 0.5)
his_d_loss = []
his_g_loss = []
his_q_loss = []
count = 0
with open("log", "a") as l:
    for epoch in range(NEPOCH):
        model.test(epoch)
        nstep = 50000//batch_size
        for step, batch_images in enumerate(train_data):
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                z, c, d = sample(batch_size)
                model.cost(batch_images, z, c, d)
                d_loss, g_loss, q_loss = model.d_loss, model.g_loss, model.mutual_info
            grad = tape.gradient(g_loss, model.G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, model.G.trainable_weights))
            grad = tape.gradient(d_loss, sum([
                model.D.get_layer('first').all_weights,
                model.D.get_layer('d1').all_weights], []))
            d_optimizer.apply_gradients(zip(grad, sum([
                model.D.get_layer('first').all_weights,
                model.D.get_layer('d1').all_weights], [])))
            grad = tape.gradient(q_loss, sum([
                model.D.get_layer('first').all_weights,
                model.D.get_layer('q1').all_weights, model.D.get_layer(
                    'q2').all_weights,
                model.D.get_layer('q3').all_weights], []))
            q_optimizer.apply_gradients(zip(grad, sum([
                model.D.get_layer('first').all_weights,
                model.D.get_layer('q1').all_weights, model.D.get_layer(
                    'q2').all_weights,
                model.D.get_layer('q3').all_weights], [])))
            del tape
            his_d_loss.append(d_loss)
            his_g_loss.append(g_loss)
            his_q_loss.append(q_loss)
            print("Epoch: [{}/{}] [{}/{}] took: {:.3f}s, d_loss: {:.5f}, g_loss: {:.5f}, q_loss: {:.5f}"
                  .format(epoch, NEPOCH, step, nstep, time.time()-step_time, d_loss, g_loss, q_loss))
            l.write("Epoch: [{}/{}] [{}/{}] took: {:.3f}s, d_loss: {:.5f}, g_loss: {:.5f}, q_loss: {:.5f}"
                    .format(epoch, NEPOCH, step, nstep, time.time()-step_time, d_loss, g_loss, q_loss)+'\n')
            count += 1
            if count % 100 == 0:
                xaxis = [i for i in range(count)]
                plt.plot(xaxis, his_d_loss)
                plt.plot(xaxis, his_g_loss)
                plt.plot(xaxis, his_q_loss)
                plt.legend(['D_Loss', 'G_Loss', 'Q_Loss'])
                plt.xlabel('Iterations')
                plt.ylabel('Loss')
                plt.savefig('results/loss.jpg')
                plt.clf()
                plt.close()
l.close()
