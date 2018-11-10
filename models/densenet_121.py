import tensorflow as tf

from .densenet_creator import DenseNetCreator


def get_model(img, classes, data_format, efficient):
    if data_format == 'channels_first':
        img = tf.transpose(img, [0, 3, 1, 2])

    return DenseNetCreator(img, classes, data_format=data_format, depth=121, efficient=efficient, nb_dense_block=4,
                           growth_rate=32, nb_filter=64, nb_layers_per_block=[6, 12, 24, 16], bottleneck=True,
                           reduction=.5, dropout_rate=0., subsample_initial_block=True, include_top=True)()
