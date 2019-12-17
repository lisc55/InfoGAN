import tensorflow as tf
import tensorlayer as tl
import imageio
import natsort
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_probability import distributions as tfd


def sample(z, size, cat=-1, c1=None, c2=None):
    if c1 is not None:
        z_con1 = np.array([c1] * size)
        z_con1 = np.reshape(z_con1, [size, 1])
    else:
        z_con1 = tfd.Uniform(low=-1.0, high=1.0).sample([size, 1])
    if c2 is not None:
        z_con2 = np.array([c2] * size)
        z_con2 = np.reshape(z_con2, [size, 1])
    else:
        z_con2 = tfd.Uniform(low=-1.0, high=1.0).sample([size, 1])
    if cat >= 0:
        z_cat = np.array([cat] * size)
        z_cat = tf.one_hot(z_cat, 10)
    else:
        z_cat = tfd.Categorical(probs=tf.ones([10])*0.1).sample([size, ])
        z_cat = tf.one_hot(z_cat, 10)
    noise = tf.concat([z, z_con1, z_con2, z_cat], axis=-1)
    return noise


generator = tl.models.Model.load('./models/model100.h5', load_weights=True)
generator.eval()

output_image = []
for i in range(10):
    z = tfd.Uniform(low=-1.0, high=1.0).sample([5, 62])
    noise = sample(z, 5, cat=i)
    imgs = generator(noise)
    imgs = (imgs + 1.) / 2.

    imgs = np.split(imgs, 5, 0)
    imgs = [np.reshape(img, [28, 28]) for img in imgs]
    imgs = np.concatenate(imgs, 0)
    output_image.append(imgs)

output_image = np.concatenate(output_image, 1)
plt.figure(figsize=(15, 8))
plt.suptitle("varying discrete latent code")
plt.imshow(output_image, cmap="gray")
plt.axis("off")
plt.savefig("./test/cat_res.png")
plt.close()

output_image = []
cc = np.linspace(-2.0, 2.0, 10)
z = tfd.Uniform(low=-1.0, high=1.0).sample([1, 62])
for i in range(5):
    imgs = []
    for ii in range(10):
        noise = sample(z, 1, cat=i, c1=cc[ii], c2=0.0)
        img = generator(noise)[0]
        img = (img + 1.) / 2.
        imgs.append(np.reshape(img, [28, 28]))
    imgs = np.concatenate(imgs, 1)
    output_image.append(imgs)

output_image = np.concatenate(output_image, 0)
plt.figure(figsize=(15, 8))
plt.suptitle("varying continuous latent code 1")
plt.imshow(output_image, cmap="gray")
plt.axis("off")
plt.savefig("./test/c1_res.png")
plt.close()

output_image = []
cc = np.linspace(-2.0, 2.0, 10)
z = tfd.Uniform(low=-1.0, high=1.0).sample([1, 62])
for i in range(5):
    imgs = []
    for ii in range(10):
        noise = sample(z, 1, cat=i, c1=-1.0, c2=cc[ii])
        img = generator(noise)[0]
        img = (img + 1.) / 2.
        imgs.append(np.reshape(img, [28, 28]))
    imgs = np.concatenate(imgs, 1)
    output_image.append(imgs)

output_image = np.concatenate(output_image, 0)
plt.figure(figsize=(15, 8))
plt.suptitle("varying continuous latent code 2")
plt.imshow(output_image, cmap="gray")
plt.axis("off")
plt.savefig("./test/c2_res.png")
plt.close()

z = tfd.Uniform(low=-1.0, high=1.0).sample([100, 62])
noise = sample(z, 100)
img = generator(noise, training=False)
img = (img + 1.) / 2.
img = tf.squeeze(img, axis=-1).numpy()
img = np.split(img, 10, 0)
img = [np.concatenate(i, 0) for i in img]
img = np.concatenate(img, 1)
plt.figure(figsize=(15, 10))
plt.suptitle("Random Generation")
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.savefig("./test/random.png")
plt.close()
