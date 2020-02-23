import tensorflow as tf
import tensorlayer as tl
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_probability import distributions as tfd
from utils import sample

tl.files.exists_or_mkdir("./test")


def sample(z, size, c1=None, c2=None, c3=None, c4=None, c5=None):
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
    if c3 is not None:
        z_con3 = np.array([c3] * size)
        z_con3 = np.reshape(z_con3, [size, 1])
    else:
        z_con3 = tfd.Uniform(low=-1.0, high=1.0).sample([size, 1])
    if c4 is not None:
        z_con4 = np.array([c4] * size)
        z_con4 = np.reshape(z_con4, [size, 1])
    else:
        z_con4 = tfd.Uniform(low=-1.0, high=1.0).sample([size, 1])
    if c5 is not None:
        z_con5 = np.array([c5] * size)
        z_con5 = np.reshape(z_con5, [size, 1])
    else:
        z_con5 = tfd.Uniform(low=-1.0, high=1.0).sample([size, 1])
    noise = tf.concat([z, z_con1, z_con2, z_con3, z_con4, z_con5], axis=-1)
    return noise


number = int(input('Model number: '))
generator = tl.models.Model.load(
    './models/model{}.h5'.format(number), load_weights=True)
generator.eval()

output_image = []
cc = np.linspace(-1, 1, 10)
z = tfd.Uniform(low=-1.0, high=1.0).sample([1, 128])
for i in range(5):
    imgs = []
    for ii in range(10):
        noise = sample(z, 1, c1=cc[ii])
        img = generator(noise)[0]
        img = (img + 1.) / 2.
        imgs.append(np.reshape(img, [32, 32]))
    imgs = np.concatenate(imgs, 1)
    output_image.append(imgs)

output_image = np.concatenate(output_image, 0)
plt.figure(figsize=(15, 8))
plt.suptitle("varying continuous latent code 1")
plt.imshow(output_image, cmap="gray")
plt.axis("off")
plt.savefig("./test/res.png")
plt.close()

z = tfd.Uniform(low=-1.0, high=1.0).sample([100, 128])
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
