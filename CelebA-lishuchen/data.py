import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from config import flags

tl.files.exists_or_mkdir(flags.checkpoint_dir)
tl.files.exists_or_mkdir(flags.result_dir)

def get_celebA(output_size, n_epoch, batch_size):
	images_path = tl.files.load_file_list(path=flags.data_dir, regx='.*.jpg', keep_prefix=True, printable=False)
	
	def generator_train():
		for image_path in images_path:
			yield image_path.encode('utf-8')

	def _map_fn(image_path):
		image = tf.io.read_file(image_path)
		image = tf.image.decode_jpeg(image, channels=3)
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		image = image[45:173, 25:153, :]
		image = tf.image.resize([image], (output_size, output_size))[0]
		return image

	ds = tf.data.Dataset.from_generator(generator_train, output_types=tf.string)
	ds = ds.shuffle(buffer_size=4096)
	ds = ds.map(_map_fn, num_parallel_calls=4)
	ds = ds.batch(batch_size)
	ds = ds.prefetch(buffer_size=2)
	return ds, images_path

if __name__ == "__main__":
	get_celebA(flags.output_size, flags.n_epoch, flags.batch_size)