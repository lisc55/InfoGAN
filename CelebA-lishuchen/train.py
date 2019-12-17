import os
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from matplotlib import pyplot as plt
from config import flags
from data import get_celebA
from model import get_G, get_DQ

def categorical_code():
	indices = np.random.randint(0, flags.dim_categorical, size=(flags.batch_size))
	return tf.one_hot(indices, flags.dim_categorical)

def gen_noise():
	noise = [categorical_code() for k in range(flags.n_categorical)]
	c = np.hstack(noise).astype(np.float32)
	noise.append(np.random.normal(size=(flags.batch_size, flags.dim_noise)))
	return np.hstack(noise).astype(np.float32), c

def kth_categorical(k):
	noise = []
	cs = []
	for i in range(flags.dim_categorical):
		cs.append(tf.one_hot(np.random.randint(0, flags.dim_categorical), flags.dim_categorical))
	cs.append(np.random.normal(size=(flags.dim_noise)))
	for i in range(flags.dim_categorical):
		cs[k] = tf.one_hot(i, flags.dim_categorical)
		noise.append(np.hstack(cs))
	return np.vstack(noise)

def gen_eval_noise():
	noise = []
	for k in range(flags.n_categorical):
		noise.append(kth_categorical(k))
	return np.vstack(noise).astype(np.float32)

def calc_mutual(output, target):
	mutual = 0
	offset = 0
	for k in range(flags.n_categorical):
		mutual += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			labels=target[:, offset : offset + flags.dim_categorical], logits=output[:, offset : offset + flags.dim_categorical]))
		offset += flags.dim_categorical
	return mutual

def train():
	images, images_path = get_celebA(flags.output_size, flags.n_epoch, flags.batch_size)
	G = get_G([None, flags.dim_z])
	D = get_DQ([None, flags.output_size, flags.output_size, flags.n_channel])

	G.train()
	D.train()

	g_optimizer = tf.optimizers.Adam(learning_rate=flags.G_learning_rate)
	d_optimizer = tf.optimizers.Adam(learning_rate=flags.D_learning_rate)
	
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
				z, c = gen_noise()
				fake_logits, fake_cat = D(G(z))
				real_logits, _ = D(batch_images)
				
				d_loss_fake = tl.cost.sigmoid_cross_entropy(output=fake_logits, target=tf.zeros_like(fake_logits), name='d_loss_fake')
				d_loss_real = tl.cost.sigmoid_cross_entropy(output=real_logits, target=tf.ones_like(real_logits), name='d_loss_real')
				d_loss = d_loss_fake + d_loss_real

				g_loss = tl.cost.sigmoid_cross_entropy(output=fake_logits, target=tf.ones_like(fake_logits), name='g_loss_fake')

				mutual = calc_mutual(fake_cat, c)
				d_loss -= mutual
				g_loss += mutual

			grad = tape.gradient(g_loss, G.trainable_weights)
			g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
			grad = tape.gradient(d_loss, D.trainable_weights)
			d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
			del tape
			print(f"Epoch: [{epoch}/{flags.n_epoch}] [{step}/{n_step_epoch}] took: {time.time()-step_time:.3f}, d_loss: {d_loss:.5f}, g_loss: {g_loss:.5f}, mutual: {mutual:.5f}")

			if count % flags.save_every_it == 1:
				his_g_loss.append(g_loss)
				his_d_loss.append(d_loss)
				his_mutual.append(mutual)

		xaxis = [i for i in range(len(his_g_loss))]
		plt.plot(xaxis, his_d_loss)
		plt.plot(xaxis, his_g_loss)
		plt.plot(xaxis, his_mutual)
		plt.legend(['D_Loss', 'G_Loss', 'Mutual_Info'])
		plt.xlabel('Iterations')
		plt.ylabel('Loss')
		plt.savefig(f'{flags.result_dir}/loss.jpg')
		plt.clf()
		plt.close()

		G.save_weights(f'{flags.checkpoint_dir}/G.npz', format='npz')
		D.save_weights(f'{flags.checkpoint_dir}/D.npz', format='npz')
		G.eval()
		z = gen_eval_noise()
		result = G(z)
		G.train()
		tl.visualize.save_images(result.numpy(), [flags.n_categorical, flags.dim_categorical], f'result/train_{epoch}.png')

if __name__ == "__main__":
	train()
