import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import flags


def get_Faces(output_size, n_epoch, batch_size):
    images_path = tl.files.load_file_list(
        path=flags.data_dir, regx='.*.png', keep_prefix=True, printable=False)

    def generator_train():
        for image_path in images_path:
            yield image_path.encode('utf-8')

    def _map_fn(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=1)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize([image], (output_size, output_size))[0]
        return image

    dataset = tf.data.Dataset.from_generator(
        generator_train, output_types=tf.string)
    dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.map(_map_fn, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=2)
    return dataset


if __name__ == "__main__":
    get_Faces(flags.output_size, flags.n_epoch, flags.batch_size)
