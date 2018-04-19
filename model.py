# This model.py file is modified from Google Tensorflow
# old version of model of rnn.translate.seq2seq_model
# You can find the original file from Google's Tensorflow
# Github repository history
# To avoid any conflict of copyright issues
# The original copyright is listed as follows:
#
# ==============================================================================
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf

import data_utils

__all__ = ["Model"]


class Model(object):
    """ Create the seq2seq base model.

    The detailed and original information of this model is described 
    in this paper: http://arxiv.org/abs/1412.7449
    """

    def __init__(self,
                 hparams,
                 buckets,
                 forward_only=False):
        """ Initialize the model.

        Args:
            hparams: the hyper-parameters of seq2seq model
            buckets: a list of buckets to improve the model efficiency
                the detailed description and instruction of the format
                of buckets can be found in original model file
            forward_only: if set, we do not construct the backward pass in
                the model. it is used to differentiate training process from
                evaluating and testing process. default value is False,
                indicating the training process
        """

        # Load model's hyper-parameters
        self.src_vocab_size = hparams.src_vocab_size
        self.tgt_vocab_size = hparams.tgt_vocab_size
        self.hidden_layer_size = hparams.hidden_layer_size
        self.num_layers = hparams.num_layers
        self.max_gradient_norm = hparams.max_gradient_norm
        self.cell_type = hparams.cell_type
        self.num_samples = hparams.num_samples
        self.batch_size = hparams.batch_size
        self.learning_rate = tf.Variable(
            float(hparams.learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * hparams.learning_rate_decay_factor)
        self.buckets = buckets
        self.global_step = tf.Variable(0, trainable=False)

        # Create placeholder for feed_dict input
        # The size placeholder depends on the largest bucket
        self.encoder_inputs = [tf.placeholder(tf.int32,
                                              shape=[None],
                                              name='encoder{0}'.format(i))
                               for i in range(self.buckets[-1][0])]
        self.decoder_inputs = [tf.placeholder(tf.int32,
                                              shape=[None],
                                              name='decoder{0}'.format(i))
                               for i in range(self.buckets[-1][1] + 1)]
        self.target_weights = [tf.placeholder(tf.float32,
                                              shape=[None],
                                              name='weight{0}'.format(i))
                               for i in range(self.buckets[-1][1] + 1)]
        # We don't want to predict the START_ID, therefore, we need to shift
        # the decoder by 1 as out target
        self.targets = [self.decoder_inputs[i + 1]
                        for i in range(len(self.decoder_inputs) - 1)]

        # Create loss function
        # To use sampled softmax, we need to construct an output projection
        # We use the same sampled softmax loss function inside original file
        # To deal with large target vocab size situation
        self.output_projection = None
        self.softmax_loss_function = None
        if self.num_samples > 0 and self.num_samples < self.tgt_vocab_size:
            w = tf.get_variable(
                'proj_w', [self.hidden_layer_size, self.tgt_vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable('proj_b', [self.tgt_vocab_size])
            self.output_projection = (w, b)

            def _sampled_softmax_loss(logits, labels):
                labels = tf.reshape(labels, [-1, 1])
                return tf.nn.sampled_softmax_loss(weights=w_t,
                                                  biases=b,
                                                  inputs=logits,
                                                  labels=labels,
                                                  num_sampled=self.num_samples,
                                                  num_classes=self.tgt_vocab_size)
            self.softmax_loss_function = _sampled_softmax_loss

        # Create cells for RNN
        if self.cell_type == 'GRU':
            single_cell = tf.contrib.rnn.GRUCell(self.hidden_layer_size)
        elif self.cell_type == 'LSTM':
            single_cell = tf.contrib.rnn.BasicLSTMCell(
                self.hidden_layer_size)
        else:
            # TODO for future different cell type
            single_cell = tf.nn.rnn_cell.GRUCell(self.hidden_layer_size)
        self.cell = tf.contrib.rnn.MultiRNNCell(
            [single_cell] * self.num_layers)

        # Create seq2seq model
        # Here we use the build-in function in tensorflow, which has
        # embedding and attention mechanism inside
        # Caution: tf.nn.seq2seq has been moved to tf.contrib.legacy_seq2seq
        # The parameter do_decode here is same as forward_only
        def seq2seq_model(encoder_inputs, decoder_inputs, do_decode):
            # Simply use the built-in model of legacy_seq2seq
            # will have a error of something as copy issue
            # It seems like the model will deepcopy the cell inside
            # However, it has no rules inside cell about how to copy itself
            # Here's a deal to this issue
            if self.cell_type == 'GRU':
                setattr(tf.contrib.rnn.GRUCell,
                        '__deepcopy__', lambda self, _: self)
            elif self.cell_type == 'LSTM':
                setattr(tf.contrib.rnn.BasicLSTMCell,
                        '__deepcopy__', lambda self, _: self)
            else:
                # TODO future different cell type
                setattr(tf.contrib.rnn.GRUCell,
                        '__deepcopy__', lambda self, _: self)
            setattr(tf.contrib.rnn.MultiRNNCell,
                    '__deepcopy__', lambda self, _: self)
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, self.cell,
                num_encoder_symbols=self.src_vocab_size,
                num_decoder_symbols=self.tgt_vocab_size,
                embedding_size=self.hidden_layer_size,
                output_projection=self.output_projection,
                feed_previous=do_decode)

        # Training procedure
        if forward_only:
            self.outputs, self.losses = \
                tf.contrib.legacy_seq2seq.model_with_buckets(
                    self.encoder_inputs,
                    self.decoder_inputs,
                    self.targets,
                    self.target_weights,
                    self.buckets,
                    lambda x, y: seq2seq_model(x, y, True),
                    softmax_loss_function=self.softmax_loss_function)
            # If we use output projection,
            # we need to project outputs for decoding
            if self.output_projection:
                for bucket_id in range(len(self.buckets)):
                    self.outputs[bucket_id] = \
                        [tf.matmul(output,
                                   self.output_projection[0]) +
                         self.output_projection[1]
                         for output in self.outputs[bucket_id]]
        else:
            self.outputs, self.losses = \
                tf.contrib.legacy_seq2seq.model_with_buckets(
                    self.encoder_inputs,
                    self.decoder_inputs,
                    self.targets,
                    self.target_weights,
                    self.buckets,
                    lambda x, y: seq2seq_model(x, y, False),
                    softmax_loss_function=self.softmax_loss_function)

        # Gradients and SGD operation
        trainable_variables = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.train_ops = []
            self.optimizer = tf.train.GradientDescentOptimizer(
                self.learning_rate)
            for bucket_id in range(len(self.buckets)):
                gradients = tf.gradients(
                    self.losses[bucket_id], trainable_variables)
                clipped_gradients, norm = \
                    tf.clip_by_global_norm(gradients,
                                           self.max_gradient_norm)
                self.gradient_norms.append(norm)
                self.train_ops.append(self.optimizer.apply_gradients(
                    zip(clipped_gradients, trainable_variables),
                    global_step=self.global_step))

        # Model saver for model saving and restoring
        self.saver = tf.train.Saver()

    def run_step(self,
                 session,
                 encoder_inputs,
                 decoder_inputs,
                 target_weights,
                 bucket_id,
                 forward_only):
        """ A single run step of the model

        Args:
            sessionL tensorflow session to use
            encoder_inputs: list of numpy int vectors to feed as encoder inputs
            decoder_inputs: list of numpy int vectors to feed as decoder inputs
            target_weights: list of numpy float vectors to feed as target weights
            bucket_id: which bucket of the model to use
            forward_only: the same idea of parameter in the model init

        Returns:
            A triple consisting of gradient norm
            (or None if we did not do backward),
            average perplexity, and the outputs.
        """

        # To whether the encoder and decoder inputs are of same size of bucket
        encoder_size, decoder_size = self.buckets[bucket_id]
        assert len(encoder_inputs) == encoder_size
        assert len(decoder_inputs) == decoder_size
        assert len(target_weights) == decoder_size

        # Input feed
        input_feed = {}
        for i in range(encoder_size):
            input_feed[self.encoder_inputs[i].name] = encoder_inputs[i]
        for i in range(decoder_size):
            input_feed[self.decoder_inputs[i].name] = decoder_inputs[i]
            input_feed[self.target_weights[i].name] = target_weights[i]

        # Since out target in model is decoder shifted by 1
        # We need to add a extra symbol at the end to stay the
        # consistency of  size, and that's the reason we need
        # one more place in decoder placeholder
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed
        if not forward_only:
            output_feed = [self.train_ops[bucket_id],
                           self.gradient_norms[bucket_id],
                           self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]
            for i in range(decoder_size):
                output_feed.append(self.outputs[bucket_id][i])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            # Gradient norm, loss, no outputs
            return outputs[1], outputs[2], None
        else:
            # No gradient norm, loss, outputs
            return None, outputs[0], outputs[1:]

    def get_batch(self,
                  data,
                  bucket_id):
        """ Get a random batch of data from a specified bucket

        This function is mainly for run_step function, to formulate
        the corresponding batch-major data

        Args:
            data: a tuple of size len(self.buckets) to create a batch
            bucket_id: integer, which bucket to get the batch for

        Returns:
            The triple (encoder_inputs, decoder_inputs, target_weights)
            for the constructed batch that has the proper format to
            call run_step function later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []

        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # Encoder inputs are padded and then reversed
            encoder_pad = [data_utils.PAD_ID] * \
                (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs are padded
            decoder_pad = [data_utils.PAD_ID] * \
                (decoder_size - len(decoder_input))
            decoder_inputs.append(decoder_input + decoder_pad)

        # Reconstruct the input to become batch major
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)],
                         dtype=np.int32))

        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)],
                         dtype=np.int32))

            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in range(self.batch_size):
                # We set weight to 0.0 if the corresponding target is padding
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or \
                        target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
