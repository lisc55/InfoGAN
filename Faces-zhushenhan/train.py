from tqdm import *
import time
import tensorlayer as tl
from data import get_Faces
from utils import sample, d_loss, g_loss, info, train_display_img
from config import flags
from model import Generator, Discriminator, Auxiliary, q_sample
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')

tl.files.exists_or_mkdir(flags.res_dir)
tl.files.exists_or_mkdir(flags.model_dir)

dataset = get_Faces(flags.output_size, flags.n_epoch, flags.batch_size)

D = Discriminator([None, 32, 32, 1])
G = Generator([None, 133])
Q = Auxiliary([None, 1024])

D.train()
G.train()
Q.train()

gen_optimizer = tf.optimizers.Adam(flags.G_learning_rate, 0.5)
dis_optimizer = tf.optimizers.Adam(flags.D_learning_rate, 0.5)


def train_step(imgs):
    noise, z_con1, z_con2, z_con3, z_con4, z_con5 = sample(flags.batch_size)
    with tf.GradientTape() as gtape, tf.GradientTape() as dtape:
        fake_imgs = G(noise)
        real_output, _ = D(imgs)
        fake_output, mid = D(fake_imgs)
        con1_mu, con1_var = Q(mid)
        fkcon1 = q_sample(con1_mu, tf.exp(con1_var))
        info_loss = info(fkcon1, z_con1)
        gen_loss = g_loss(fake_output)
        dis_loss = d_loss(real_output, fake_output)
        gi = gen_loss+flags.info_lambda*info_loss
        di = dis_loss

    g_grd = gtape.gradient(gi, G.trainable_weights+Q.trainable_weights)
    d_grd = dtape.gradient(di, D.trainable_weights)
    gen_optimizer.apply_gradients(
        zip(g_grd, G.trainable_weights+Q.trainable_weights))
    dis_optimizer.apply_gradients(zip(d_grd, D.trainable_weights))

    return gen_loss, dis_loss, info_loss


def train(dataset, epochs):
    step = 0
    gen_loss = []
    dis_loss = []
    info_loss = []
    for epoch in range(epochs):
        for batch in tqdm(dataset):
            gen, dis, info = train_step(batch)
            gen_loss.append(gen)
            dis_loss.append(dis)
            info_loss.append(info)
            step += 1
            if step % 100 == 0:
                train_display_img(G, step)

        G.save("./models/model{}.h5".format(epoch+1), save_weights=True)
        mg = tf.reduce_mean(gen_loss).numpy()
        md = tf.reduce_mean(dis_loss).numpy()
        mi = tf.reduce_mean(info_loss).numpy()
        print("[{}]\t{:03d}\tGenerator: {:.4f}\tDiscriminator: {:.4f}\tInfo: {:.4f}".format(
            time.strftime('%H:%M:%S', time.localtime(time.time())), epoch+1, mg, md, mi))

    plt.figure(figsize=(20, 8))
    plt.plot(gen_loss, label="generator")
    plt.plot(dis_loss, label="discriminator")
    plt.plot(info_loss, label="mutual_info")
    plt.legend()
    plt.suptitle("GAN loss")
    plt.savefig("loss")
    plt.close()


train(dataset, flags.n_epoch)
