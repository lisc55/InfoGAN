import os, time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
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

def train():
	images, images_path = get_celebA(flags.output_size, flags.n_epoch, flags.batch_size)
	G = get_G([None, flags.dim_z])
	D = get_DQ([None, flags.output_size, flags.output_size, flags.n_channel])

	G.train()
	D.train()

	g_optimizer = tf.optimizers.Adam(learning_rate=flags.G_learning_rate)
	d_optimizer = tf.optimizers.Adam(learning_rate=flags.D_learning_rate)
	
	n_step_epoch = int(len(images_path) // flags.batch_size)

	for epoch in range(flags.n_epoch):
		for step, batch_images in enumerate(images):
			if batch_images.shape[0] != flags.batch_size:
				break
			step_time = time.time()
			with tf.GradientTape(persistent=True) as tape:
				z, c = gen_noise()
				fake_logits, fake_cat = D(G(z))
				real_logits, _ = D(batch_images)
				
				d_loss_fake = tl.cost.sigmoid_cross_entropy(fake_logits, tf.zeros_like(fake_logits), name='d_loss_fake')
				d_loss_real = tl.cost.sigmoid_cross_entropy(real_logits, tf.ones_like(real_logits), name='d_loss_real')
				d_loss = d_loss_fake + d_loss_real

				g_loss = tl.cost.sigmoid_cross_entropy(fake_logits, tf.ones_like(fake_logits), name='g_loss_fake')

				q_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=c, logits=fake_cat))
				d_loss += q_loss
				g_loss += q_loss
			
			grad = tape.gradient(g_loss, G.trainable_weights)
			g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
			grad = tape.gradient(d_loss, D.trainable_weights)
			d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
			del tape
			print(f"Epoch: [{epoch}/{flags.n_epoch}] [{step}/{n_step_epoch}] took: {time.time()-step_time:.3f}, d_loss: {d_loss:.5f}, g_loss: {g_loss:.5f}, mutual_info: {q_loss:.5f}")
		
		G.save_weights(f'{flags.checkpoint_dir}/G.npz', format='npz')
		D.save_weights(f'{flags.checkpoint_dir}/D.npz', format='npz')
		G.eval()
		result = G(z)
		G.train()
		tl.visualize.save_images(result.numpy(), [8, 8], 'train_{:02d}.png'.format(epoch))

if __name__ == "__main__":
	train()