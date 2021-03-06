import os
import time
import numpy as np
np.set_printoptions(threshold=np.inf, precision=1, linewidth=np.inf)
import tensorflow as tf
import tensorlayer as tl
from matplotlib import pyplot as plt
from config import flags
from data import get_celebA
from model import get_G, get_base, get_D, get_Q

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

def gen_eval_noise(k, n_sample):
	return np.vstack([kth_categorical(k) for i in range(n_sample)]).astype(np.float32)

def calc_mutual(output, target):
	mutual = 0
	offset = 0
	for k in range(flags.n_categorical):
		mutual += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			labels=target[:, offset : offset + flags.dim_categorical], logits=output[:, offset : offset + flags.dim_categorical]))
		offset += flags.dim_categorical
	return mutual

def convert(images):
	for i in range(len(images)):
		mi = np.min(images[i])
		mx = np.max(images[i])
		images[i] = (images[i] - mi) / (mx - mi)
	return images

def train():
	images, images_path = get_celebA(flags.output_size, flags.n_epoch, flags.batch_size)
	G = get_G([None, flags.dim_z])
	Base = get_base([None, flags.output_size, flags.output_size, flags.n_channel])
	D = get_D([None, 4096])
	Q = get_Q([None, 4096])

	G.train()
	Base.train()
	D.train()
	Q.train()

	g_optimizer = tf.optimizers.Adam(learning_rate=flags.G_learning_rate, beta_1=flags.beta_1)
	d_optimizer = tf.optimizers.Adam(learning_rate=flags.D_learning_rate, beta_1=flags.beta_1)
	
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
				fake = Base(G(z))
				fake_logits = D(fake)
				fake_cat = Q(fake)
				real_logits = D(Base(batch_images))
				
				d_loss_fake = tl.cost.sigmoid_cross_entropy(output=fake_logits, target=tf.zeros_like(fake_logits), name='d_loss_fake')
				d_loss_real = tl.cost.sigmoid_cross_entropy(output=real_logits, target=tf.ones_like(real_logits), name='d_loss_real')
				d_loss = d_loss_fake + d_loss_real

				g_loss = tl.cost.sigmoid_cross_entropy(output=fake_logits, target=tf.ones_like(fake_logits), name='g_loss_fake')

				mutual = calc_mutual(fake_cat, c)
				g_loss += mutual
				
			grad = tape.gradient(g_loss, G.trainable_weights + Q.trainable_weights)
			g_optimizer.apply_gradients(zip(grad, G.trainable_weights + Q.trainable_weights))
			grad = tape.gradient(d_loss, D.trainable_weights + Base.trainable_weights)
			d_optimizer.apply_gradients(zip(grad, D.trainable_weights + Base.trainable_weights))
			del tape
			print(f"Epoch: [{epoch}/{flags.n_epoch}] [{step}/{n_step_epoch}] took: {time.time()-step_time:.3f}, d_loss: {d_loss:.5f}, g_loss: {g_loss:.5f}, mutual: {mutual:.5f}")

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
		for k in range(flags.n_categorical):
			z = gen_eval_noise(k, flags.n_sample)
			result = G(z)
			tl.visualize.save_images(convert(result.numpy()), [flags.n_sample, flags.dim_categorical], f'result/train_{epoch}_{k}.png')
		G.train()

if __name__ == "__main__":
	train()
