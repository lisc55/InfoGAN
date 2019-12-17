import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from model import Generator, Discriminator, Auxiliary, q_sample
from config import flags
from utils import sample, d_loss, g_loss, info, train_display_img
import tensorlayer as tl
import time

(Xtr, ytr), (_, _) = tf.keras.datasets.mnist.load_data()
Xtr = (Xtr/127.5)-1
Xtr = Xtr.reshape([-1, 28, 28, 1]).astype("float32")
dataset = tf.data.Dataset.from_tensor_slices(Xtr)
dataset = dataset.shuffle(30000).batch(flags.batch_size)

D = Discriminator([None, 28, 28, 1])
G = Generator([None, 74])
Q = Auxiliary([None, 1024])

D.train()
G.train()
Q.train()

gen_optimizer = tf.keras.optimizers.Adam(flags.G_learning_rate)
dis_optimizer = tf.keras.optimizers.Adam(flags.D_learning_rate)


def train_step(imgs):
    noise, z_con1, z_con2, z_cat = sample(flags.batch_size)
    with tf.GradientTape() as gtape, tf.GradientTape() as dtape:
        fake_imgs = G(noise)
        real_output, _ = D(imgs)
        fake_output, mid = D(fake_imgs)
        cat, con1_mu, con1_var, con2_mu, con2_var = Q(mid)
        fkcat, fkcon1, fkcon2 = cat, q_sample(
            con1_mu, tf.exp(con1_var)), q_sample(con2_mu, tf.exp(con2_var))
        info_loss, c1, c2, sce = info(
            fkcon1, fkcon2, fkcat, z_con1, z_con2, z_cat)
        gen_loss = g_loss(fake_output)
        dis_loss = d_loss(real_output, fake_output)
        gi = gen_loss+info_loss
        di = dis_loss+info_loss

    g_grd = gtape.gradient(gi, G.trainable_weights+Q.trainable_weights)
    d_grd = dtape.gradient(di, D.trainable_weights)
    gen_optimizer.apply_gradients(zip(g_grd, G.trainable_weights))
    dis_optimizer.apply_gradients(zip(d_grd, D.trainable_weights))

    return gen_loss, dis_loss, info_loss


def train(dataset, epochs):
    step = 0
    gen_loss = []
    dis_loss = []
    info_loss = []
    for epoch in range(epochs):
        for batch in dataset:
            gen, dis, info = train_step(batch)
            gen_loss.append(gen)
            dis_loss.append(dis)
            info_loss.append(info)
            mg = tf.reduce_mean(gen_loss).numpy()
            md = tf.reduce_mean(dis_loss).numpy()
            mi = tf.reduce_mean(info_loss).numpy()
            print("[{}]    {:05d}/{:03d}    Generator: {:.4f}    Discriminator: {:.4f}    Info: {:.4f}".format(
                time.strftime('%H:%M:%S', time.localtime(time.time())), step % 60000+1, epoch+1, mg, md, mi))
            step += 1
            if step % 100 == 0:
                train_display_img(G, step)

        G.save("./models/model{}.h5".format(epoch+1), save_weights=True)

    plt.figure(figsize=(20, 8))
    plt.plot(gen_loss, label="generator")
    plt.plot(dis_loss, label="discriminator")
    plt.plot(info_loss, label="mutual_info")
    plt.legend()
    plt.suptitle("GAN loss")
    plt.savefig("loss")


train(dataset, flags.n_epoch)
