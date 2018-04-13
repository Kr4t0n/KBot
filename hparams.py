from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import namedtuple

"""

    self.src_vocab_size = hparams.src_vocab_size
    self.tgt_vocab_size = hparams.tgt_vocab_size
"""


# Model Hyperparameter
tf.flags.DEFINE_integer("hidden_layer_size", 512,
                        "Number of cells in each layer")
tf.flags.DEFINE_integer("num_layers", 4, "Number of layers in model")
tf.flags.DEFINE_string("cell_type", "LSTM", "Type of RNN cell")

tf.flags.DEFINE_float("learning_rate", 0.5, "Learning rate")
tf.flags.DEFINE_float("learning_rate_decay_factor",
                      0.99, 'Learning rate decays by this much.')
tf.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size during training")
tf.flags.DEFINE_integer(
    "num_samples", 512, "Number of samples to calculate loss")

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
    "HParams",
    [
        "batch_size",
        "cell_type",
        "hidden_layer_size",
        "learning_rate",
        "learning_rate_decay_factor",
        "max_gradient_norm",
        "num_layers",
        "num_samples"
    ])


def create_hparams():
    return HParams(
        batch_size=FLAGS.batch_size,
        cell_type=FLAGS.cell_type,
        hidden_layer_size=FLAGS.hidden_layer_size,
        learning_rate=FLAGS.learning_rate,
        learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
        max_gradient_norm=FLAGS.max_gradient_norm,
        num_layers=FLAGS.num_layers,
        num_samples=FLAGS.num_samples)
