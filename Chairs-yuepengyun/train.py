import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import time
from matplotlib import pyplot as plt
from data import get_Chairs, flags
from model import get_D, get_G, get_Q


def gen_noise():
    indice1 = np.random.randint(
        low=0, high=flags.dim_categorical, size=flags.batch_size)
    indice2 = np.random.randint(
        low=0, high=flags.dim_categorical, size=flags.batch_size)
    indice3 = np.random.randint(
        low=0, high=flags.dim_categorical, size=flags.batch_size)
    cat1 = tf.one_hot(indice1, flags.dim_categorical)
    cat2 = tf.one_hot(indice2, flags.dim_categorical)
    cat3 = tf.one_hot(indice3, flags.dim_categorical)
    con = np.random.rand(flags.batch_size, 1)
    con = con*2.0-1.0
    con = con.astype(np.float32)
    z = np.random.normal(loc=0.0, scale=1.0, size=[
                         flags.batch_size, flags.dim_noise])
    noise = tf.concat([con, cat1, cat2, cat3, z], axis=-1)
    return noise, cat1, cat2, cat3, tf.convert_to_tensor(con)


def gen_eval_noise(cases, stats):
    noise = []
    for i in range(cases):
        indice1 = np.random.randint(
            low=0, high=flags.dim_categorical)
        indice2 = np.random.randint(
            low=0, high=flags.dim_categorical)
        indice3 = np.random.randint(
            low=0, high=flags.dim_categorical)
        cat1 = tf.convert_to_tensor(
            [tf.one_hot(indice1, flags.dim_categorical) for i in range(stats)])
        cat2 = tf.convert_to_tensor(
            [tf.one_hot(indice2, flags.dim_categorical) for i in range(stats)])
        cat3 = tf.convert_to_tensor(
            [tf.one_hot(indice3, flags.dim_categorical) for i in range(stats)])
        con = np.ones([flags.n_samples, 1], np.float32)
        for i in range(stats):
            con[i] *= (i/float(stats-1))
        con = tf.convert_to_tensor(con*2.0-1.0)
        z = np.random.normal(loc=0.0, scale=1.0,
                             size=flags.dim_noise).astype(np.float32)
        z = tf.convert_to_tensor([z for i in range(stats)])
        temp_noise = tf.concat([con, cat1, cat2, cat3, z], axis=-1)
        if len(noise) == 0:
            noise = temp_noise
        else:
            noise = tf.concat([noise, temp_noise], axis=0)
    return noise


def calc_disc_mutual(f_cat1, f_cat2, f_cat3, cat1, cat2, cat3):
    sce = tf.keras.losses.categorical_crossentropy(y_true=cat1, y_pred=f_cat1,
                                                   from_logits=True)
    sce += tf.keras.losses.categorical_crossentropy(y_true=cat2, y_pred=f_cat2,
                                                    from_logits=True)
    sce += tf.keras.losses.categorical_crossentropy(y_true=cat3, y_pred=f_cat3,
                                                    from_logits=True)
    return tf.reduce_mean(sce)


def calc_cont_mutual(f_con, con):
    c1 = tf.reduce_mean(tf.square(f_con-con))
    return c1


def train():
    images, images_path = get_Chairs(
        flags.output_size, flags.n_epoch, flags.batch_size)
    G = get_G([None, flags.z_dim])
    D = get_D([None, flags.output_size, flags.output_size, flags.n_channel])
    Q = get_Q([None, 1024])

    G.train()
    D.train()
    Q.train()

    g_optimizer = tf.optimizers.Adam(
        learning_rate=flags.G_learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(
        learning_rate=flags.D_learning_rate, beta_1=0.5)

    n_step_epoch = int(len(images_path) // flags.batch_size)
    his_g_loss = []
    his_d_loss = []
    his_mutual = []
    count = 0

    for epoch in range(flags.n_epoch):
        for step, batch_images in enumerate(images):
            count += 1
            if batch_images.shape[0] != flags.batch_size:
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                noise, cat1, cat2, cat3, con = gen_noise()
                fake_logits, mid = D(G(noise))
                real_logits, _ = D(batch_images)
                f_cat1, f_cat2, f_cat3, f_mu = Q(mid)

                # base = tf.random.normal(shape=f_mu.shape)
                # f_con = f_mu + base * tf.exp(f_sigma)
                d_loss_fake = tl.cost.sigmoid_cross_entropy(
                    output=fake_logits, target=tf.zeros_like(fake_logits), name='d_loss_fake')
                d_loss_real = tl.cost.sigmoid_cross_entropy(
                    output=real_logits, target=tf.ones_like(real_logits), name='d_loss_real')
                d_loss = d_loss_fake + d_loss_real

                g_loss_tmp = tl.cost.sigmoid_cross_entropy(
                    output=fake_logits, target=tf.ones_like(fake_logits), name='g_loss_fake')

                mutual_disc = calc_disc_mutual(
                    f_cat1, f_cat2, f_cat3, cat1, cat2, cat3)
                mutual_cont = calc_cont_mutual(f_mu, con)
                mutual = (flags.disc_lambda*mutual_disc +
                          flags.cont_lambda*mutual_cont)
                g_loss = mutual + g_loss_tmp
                d_tr = d_loss + mutual

            grads = tape.gradient(
                g_loss, G.trainable_weights + Q.trainable_weights)  # 一定要可求导
            g_optimizer.apply_gradients(
                zip(grads, G.trainable_weights + Q.trainable_weights))
            grads = tape.gradient(
                d_tr, D.trainable_weights)
            d_optimizer.apply_gradients(
                zip(grads, D.trainable_weights))
            del tape

            print("Epoch: [{}/{}] [{}/{}] took: {}, d_loss: {:.5f}, g_loss: {:.5f}, mutual: {:.5f}".format(
                epoch, flags.n_epoch, step, n_step_epoch, time.time()-step_time, d_loss, g_loss, mutual))

            if count % flags.save_every_it == 1:
                his_g_loss.append(g_loss)
                his_d_loss.append(d_loss)
                his_mutual.append(mutual)

        plt.plot(his_d_loss)
        plt.plot(his_g_loss)
        plt.plot(his_mutual)
        plt.legend(['D_Loss', 'G_Loss', 'Mutual_Info'])
        plt.xlabel(f'Iterations / {flags.save_every_it}')
        plt.ylabel('Loss')
        plt.savefig(f'{flags.result_dir}/loss.jpg')
        plt.clf()
        plt.close()

        G.save_weights(f'{flags.checkpoint_dir}/G.npz', format='npz')
        D.save_weights(f'{flags.checkpoint_dir}/D.npz', format='npz')
        G.eval()
        for k in range(flags.n_samples):
            z = gen_eval_noise(flags.save_every_epoch, flags.n_samples)
            result = G(z)
            tl.visualize.save_images(result.numpy(), [
                                     flags.save_every_epoch, flags.n_samples], f'result/train_{epoch}_{k}.png')
        G.train()


if __name__ == '__main__':
    train()
