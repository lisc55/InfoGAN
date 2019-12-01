import tensorflow as tf


def load_mnist_data(batch_size=64):
    (train_images, train_labels), (test_images,
                                   test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1).astype('float32')
    BUFFER_SIZE = train_images.shape[0]
    train_images = (train_images)/255.0
    train_labels = tf.one_hot(train_labels, depth=10)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
        BUFFER_SIZE).batch(batch_size, drop_remainder=True)
    return train_dataset
