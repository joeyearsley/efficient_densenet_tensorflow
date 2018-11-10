"""Runs a ResNet model on the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from constants import HEIGHT, WIDTH, DEPTH, NUM_CLASSES, NUM_IMAGES, MOMENTUM, WEIGHT_DECAY
from dataset import input_fn
from models import densenet_121
from utils import float32_variable_storage_getter

import argparse
import os
import sys
import horovod.tensorflow as hvd

import tensorflow as tf

from tensorflow.contrib.mixed_precision import ExponentialUpdateLossScaleManager, LossScaleOptimizer


parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='The path to the CIFAR-10 data directory.')

parser.add_argument('--model_dir', type=str, default='/tmp/cifar10_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--train_epochs', type=int, default=250,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=10,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=3750,
                    help='The number of images per batch.')

parser.add_argument('--fp16', type=bool, default=False,
                    help='Whether to run with FP16 or not.')

parser.add_argument('--efficient', type=bool, default=False,
                    help='Whether to run with gradient checkpointing or not.')

parser.add_argument(
    '--data_format', type=str, default='channels_first',
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')


def cifar10_model_fn(features, labels, params):
    print('PARAMS', params['fp16'])
    """Model function for CIFAR-10."""
    tf.summary.image('images', features, max_outputs=6)

    inputs = tf.reshape(features, [-1, HEIGHT, WIDTH, DEPTH])
    if params['fp16']:
        inputs = tf.cast(inputs, tf.float16)

    logits = densenet_121.get_model(inputs, NUM_CLASSES, params['data_format'], params['efficient'])
    logits = tf.cast(logits, tf.float32)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss.
    loss = cross_entropy + WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    # Scale the learning rate linearly with the batch size. When the batch size
    # is 128, the learning rate should be 0.1.
    initial_learning_rate = 0.1 * params['batch_size'] / 128
    batches_per_epoch = NUM_IMAGES['train'] / params['batch_size']
    global_step = tf.train.get_or_create_global_step()

    # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
    values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=MOMENTUM)

    if params['fp16']:
        # Choose a loss scale manager which decides how to pick the right loss scale
        # throughout the training process.
        loss_scale_manager = ExponentialUpdateLossScaleManager(128, 100)
        # Wraps the original optimizer in a LossScaleOptimizer.
        optimizer = LossScaleOptimizer(optimizer, loss_scale_manager)

    compression = hvd.Compression.fp16 if params['fp16'] else hvd.Compression.none

    optimizer = hvd.DistributedOptimizer(optimizer, compression=compression)

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)

    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])
    metrics = {'accuracy': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy', accuracy[1])

    return train_op, loss, global_step


def main(unused_argv):
    # Initialize Horovod.
    hvd.init()

    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    custom_params = {
        'data_format': FLAGS.data_format,
        'batch_size': FLAGS.batch_size,
        'fp16': FLAGS.fp16,
        'efficient': FLAGS.efficient
    }

    features, labels = input_fn(True, FLAGS.data_dir, FLAGS.batch_size, None)
    with tf.variable_scope('model', custom_getter=float32_variable_storage_getter):
        train_op, loss, global_step = cifar10_model_fn(features, labels, custom_params)

    # BroadcastGlobalVariablesHook broadcasts initial variable states from rank 0
    # to all other processes. This is necessary to ensure consistent initialization
    # of all workers when training is started with random weights or restored
    # from a checkpoint.
    hooks = [hvd.BroadcastGlobalVariablesHook(0),
             tf.train.StopAtStepHook(last_step=10000),
             tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss},
                                        every_n_iter=10),
             ]

    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Save checkpoints only on worker 0 to prevent other workers from corrupting them.
    checkpoint_dir = './checkpoints' if hvd.rank() == 0 else None

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        while not mon_sess.should_stop():
            # Run a training step synchronously.
            mon_sess.run(train_op)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(argv=[sys.argv[0]] + unparsed)
