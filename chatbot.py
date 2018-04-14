from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import time
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from . import model
from . import data_utils
from . import chatbot_hparams


def create_seq2seq_model(session, hparams, forword_only):
    seq2seq_model = model.Model(
        hparams=hparams,
        buckets=data_utils.BUCKETS,
        forword_only=forword_only)

    ckpt = tf.train.get_checkpoint_state(data_utils.CPT_DIR)
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


def get_hparams():
    hparams = chatbot_hparams.create_hparams
    src_vocab_size = data_utils.get_vocab_size(
        data_utils.DATA_PATH + 'vocab.in')
    tgt_vocab_size = data_utils.get_vocab_size(
        data_utils.DATA_PATH + 'vocab.ou')
    hparams.add_hparam("src_vocab_size", src_vocab_size)
    hparams.add_hparam("tgt_vocab_size", tgt_vocab_size)

    return hparams


def train(hparams):
    with tf.Session() as sess:
        model = create_seq2seq_model(sess, hparams, forword_only=False)

        train_sets = data_utils.load_data('train_ids.in', 'train_ids.ou')
        dev_sets = data_utils.load_data('dev_ids.in', 'dev_ids.ou')
        train_bucket_sizes = [len(train_sets[bucket_id])
                              for bucket_id in range(len(data_utils.BUCKETS))]
        train_total_size = sum(train_bucket_sizes)
        train_bucket_scale = [sum(train_bucket_sizes[:i + 1]) /
                              train_total_size
                              for i in range(len(train_bucket_sizes))]

        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        while True:
            rand = random.random()
            bucket_id = min([i for i in range(len(train_bucket_scale))
                             if train_bucket_scale[i] > rand])

            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = \
                model.get_batch(train_sets, bucket_id)
            _, step_loss, _ = model.run_step(session=sess,
                                             encoder_inputs=encoder_inputs,
                                             decoder_inputs=decoder_inputs,
                                             target_weights=target_weights,
                                             bucket_id=bucket_id,
                                             forword_only=False)
            step_time += time.time() - start_time
            loss += step_loss / hparams.steps_per_checkpoint
            current_step += 1

            if current_step % hparams.steps_per_checkpoint == 0:
                print("Global step %d: loss %.6f, learning rate %.4f, step time %.2f" %
                      (loss,
                       model.global_step.eval(),
                       model.learning_rate.eval(),
                       time.time() - start_time))

                if len(previous_losses) > 2 and \
                        loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)

                cpt_path = os.path.join(data_utils.CPT_DIR + "chatbot")
                model.saver.save(sess, cpt_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                for bucket_id in range(len(data_utils.BUCKETS)):
                    encoder_inputs, decoder_inputs, target_weights = \
                        model.get_batch(dev_sets, bucket_id)
                    _, eval_loss, _ = \
                        model.run_step(session=sess,
                                       encoder_inputs=encoder_inputs,
                                       decoder_inputs=decoder_inputs,
                                       target_weights=target_weights,
                                       bucket_id=bucket_id,
                                       forward_only=True)
                    print("Eval: bucket %d, loss %.6f" %
                          (bucket_id, eval_loss))

                    sys.stdout.flush()


def chat(hparams):
    with tf.Session() as sess:
        model = create_seq2seq_model(sess, hparams, forword_only=True)
        model.batch_size = 1

        src_vocab, _ = data_utils.initialize_vocab("vocab.in")
        _, rev_vocab = data_utils.initialize_vocab("vocab.ou")

        print("Hello! I'm a chatbot! Say something!")
        while True:
            print("> ", end='')
            sys.stdout.flush()
            sentence = sys.stdin.readline()

            if len(sentence) > 0 and sentence[-1] == '\n':
                sentence = sentence[:-1]
            if sentence == '':
                break
            response = get_predicted_sentence(session=sess,
                                              input_sentence=sentence,
                                              src_vocab=src_vocab,
                                              rev_vocab=rev_vocab,
                                              seq2seq_model=model)
            print(response)


def chatbot():
    hparams = get_hparams()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices={'train', 'chat'}, default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train(hparams)
    elif args.mode == 'chat':
        chat(hparams)


if __name__ == '__main__':
    chatbot()
