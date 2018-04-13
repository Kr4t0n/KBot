from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from . import model
from . import config
from . import data_utils


def create_seq2seq_model(session, hparams, forword_only):
    seq2seq_model = model.Model(
        hparams=hparams,
        buckets=data_utils.BUCKETS,
        forword_only=forword_only)

    ckpt = tf.train.get_checkpoint_state(config.CPT_DIR)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        print("Loading model parameters from {}...".format(
            ckpt.model_checkpoint_path))
        seq2seq_model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Creating model from scratch...")
        session.run(tf.global_variables_initializer())

    return seq2seq_model


def get_predicted_sentence(session,
                           input_sentence,
                           src_vocab,
                           rev_vocab,
                           seq2seq_model):
    input_token_ids = data_utils.sentence_to_token_ids(
        input_sentence, src_vocab)

    bucket_id = min([b for b in range(len(data_utils.BUCKETS))
                     if data_utils.BUCKETS[b][0] > len(input_token_ids)])
    outputs = []
    feed_data = {bucket_id: [(input_token_ids, outputs)]}

    # Construct the batch of this element to feed the model
    encoder_inputs, decoder_inputs, target_weights = \
        seq2seq_model.get_batch(feed_data, bucket_id)

    # Get output logits from the model
    _, _, output_logits = seq2seq_model.run_step(
        session=session,
        encoder_inputs=encoder_inputs,
        decoder_inputs=decoder_inputs,
        target_weights=target_weights,
        bucket_id=bucket_id,
        forward_only=True)

    responses = []
    # Greedy decoder
    for logit in output_logits:
        selected_token_id = int(np.argmax(logit, axis=1))

        if selected_token_id == data_utils.END_ID:
            break
        else:
            responses.append(selected_token_id)

    output_sentence = ' '.join([rev_vocab[response] for response in responses])

    return output_sentence
