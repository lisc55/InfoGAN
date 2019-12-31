import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl


class FLAGS(object):
    def __init__(self):
        self.n_epoch = 100
        self.D_learning_rate = 0.0002
        self.G_learning_rate = 0.001
        self.leaky_rate = 0.1
        self.n_categorical = 3  # 3 discrete latent codes
        self.dim_categorical = 20  # each with dimension 20
        self.c_dim = 61
        self.cont_lambda = 10.0
        self.disc_lambda = 1.0
        self.dim_noise = 128
        self.z_dim = 189
        self.batch_size = 128
        self.output_size = 64  # ?
        self.n_channel = 1
        self.n_samples = 10
        self.save_every_epoch = 6
        self.save_every_it = 50
        self.data_dir = "/data2/lishuchen/rendered_chairs"
        self.checkpoint_dir = "checkpoint"
        self.result_dir = "result"


flags = FLAGS()

tl.files.exists_or_mkdir(flags.checkpoint_dir)
tl.files.exists_or_mkdir(flags.result_dir)


def get_Chairs(output_size, n_epoch, batch_size):
    folders_path1 = tl.files.load_folder_list(path=flags.data_dir)
    folders_path2 = []
    for folder in folders_path1:
        folders_path2 += tl.files.load_folder_list(path=folder)
    images_path = []
    for folder in folders_path2:
        images_path += tl.files.load_file_list(
            path=folder, regx='.*.png', keep_prefix=True, printable=False)

    def generator_train():
        for image_path in images_path:
            yield image_path.encode('utf-8')

    def _map_fn(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=1)  # ???
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = image[120:480, 120:480, :]  # ???
        image = tf.image.resize([image], (output_size, output_size))[0]
        return image

    ds = tf.data.Dataset.from_generator(
        generator_train, output_types=tf.string)
    ds = ds.shuffle(buffer_size=4096)
    ds = ds.map(_map_fn, num_parallel_calls=4)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=2)
    return ds, images_path


if __name__ == "__main__":
    get_Chairs(flags.output_size, flags.n_epoch, flags.batch_size)
