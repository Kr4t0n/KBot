from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random
import time
import numpy as np
import tensorflow as tf

import model
import data_utils
import chatbot_hparams

# The functions,  create_seq2seq_model and get_predicted_sentence
# are modified from Google's seq2seq translate project
# You can find original functions from seq2seq_model_utils.py file
# The functions, train and chat are also modified for the need of
# chatbot


def create_seq2seq_model(session, hparams, forward_only):
    """ Create seq2seq model and load model hyper parameters into it
    """
    seq2seq_model = model.Model(
        hparams=hparams,
        buckets=data_utils.BUCKETS,
        forward_only=forward_only)

    # We load checkpoint here since every time we create a model
    # we need to load checkpoint once
    ckpt = tf.train.get_checkpoint_state(
        os.path.dirname(data_utils.CPT_DIR + 'checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
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
    """ Create one single step to get response from model
    """
    input_token_ids = data_utils.sentence_to_token_ids(
        input_sentence, src_vocab)

    # To get the correct bucket_id for this sentence
    bucket_id = min([b for b in range(len(data_utils.BUCKETS))
                     if data_utils.BUCKETS[b][0] > len(input_token_ids)])
    # Since we need to obtain the response, so we do not need any
    # read decoder_input here
    decoder = []
    feed_data = {bucket_id: [(input_token_ids, decoder)]}

    # Construct the batch of this element to feed the model
    encoder_inputs, decoder_inputs, target_weights = \
        seq2seq_model.get_batch(feed_data, bucket_id)

    # Get output logits from the model
    # We only need logits here
    _, _, output_logits = seq2seq_model.run_step(
        session=session,
        encoder_inputs=encoder_inputs,
        decoder_inputs=decoder_inputs,
        target_weights=target_weights,
        bucket_id=bucket_id,
        forward_only=True)

    responses = []
    # Greedy decoder here
    # TODO further beam search decoder can add here
    for logit in output_logits:
        # Each time, we just find the most possible word here
        selected_token_id = int(np.argmax(logit, axis=1))

        if selected_token_id == data_utils.END_ID:
            break
        else:
            responses.append(selected_token_id)

    output_sentence = ' '.join([rev_vocab[response] for response in responses])

    return output_sentence


def get_hparams():
    """ Load model hyper parameters from chatbot_hparams.py file
    """
    hparams = chatbot_hparams.create_hparams()
    # The size of vocab here is dynamically add to hparams
    # in order to avoid change the size manually when using different
    # data or vocab
    src_vocab_size = data_utils.get_vocab_size('vocab.in')
    tgt_vocab_size = data_utils.get_vocab_size('vocab.ou')
    hparams.add_hparam("src_vocab_size", src_vocab_size)
    hparams.add_hparam("tgt_vocab_size", tgt_vocab_size)

    return hparams


def train(hparams):
    """ The main function of train process
    """
    with tf.Session() as sess:
        # Since it is the train process, we need to set forward_only as False
        # cause we need backward to refine our model params
        model = create_seq2seq_model(session=sess,
                                     hparams=hparams,
                                     forward_only=False)

        # Load train sets and dev sets
        train_sets = data_utils.load_data('train_ids.in', 'train_ids.ou')
        dev_sets = data_utils.load_data('dev_ids.in', 'dev_ids.ou')
        train_bucket_sizes = [len(train_sets[bucket_id])
                              for bucket_id in range(len(data_utils.BUCKETS))]
        train_total_size = sum(train_bucket_sizes)
        # The idea of scale here is that each time we choose a random bucket
        # for training, the chance of picking a particular bucket is
        # proportional to the size of that bucket
        train_bucket_scale = [sum(train_bucket_sizes[:i + 1]) /
                              train_total_size
                              for i in range(len(train_bucket_sizes))]

        run_step_time, losses = 0.0, 0.0
        previous_losses = []

        while True:
            rand = random.random()
            bucket_id = min([i for i in range(len(train_bucket_scale))
                             if train_bucket_scale[i] > rand])

            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = \
                model.get_batch(train_sets, bucket_id)
            # We only need step_loss for statistic purpose
            _, step_loss, _ = model.run_step(session=sess,
                                             encoder_inputs=encoder_inputs,
                                             decoder_inputs=decoder_inputs,
                                             target_weights=target_weights,
                                             bucket_id=bucket_id,
                                             forward_only=False)
            run_step_time += time.time() - start_time
            losses += step_loss / hparams.steps_per_checkpoint

            # For every steps_per_checkpoint, we need to print out the
            # current state of model, and save the model in checkpoint folder
            if model.global_step.eval() % hparams.steps_per_checkpoint == 0:
                print("Global step {}: loss {}, learning rate {}, step time {}".format(
                    model.global_step.eval(),
                    losses,
                    model.learning_rate.eval(),
                    run_step_time))

                # Consider whether we should decay the learning rate or not
                # If the losses get higher than previous losses
                # We need to decay our learning rate a little bit
                if len(previous_losses) > 2 and \
                        losses > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(losses)

                cpt_path = os.path.join(data_utils.CPT_DIR + "chatbot")
                model.saver.save(sess, cpt_path, global_step=model.global_step)
                run_step_time, losses = 0.0, 0.0

                # For every 10 * steps_per_checkpoint, we evaluate the model
                # using dev_sets, and print out the evaluation losses
                # The reason that forward_only is True here, cause we do not
                # want use dev_sets to refine out model params
                if model.global_step.eval() % \
                        (10 * hparams.steps_per_checkpoint) == 0:
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
                        print("Eval: bucket {}: loss {}".format(
                            bucket_id, eval_loss))
                        sys.stdout.flush()


def chat(hparams):
    """ The main function of chat process, to chat with chatbot
    """
    with tf.Session() as sess:
        # Since it is not train process, we need to set forward_only as True
        model = create_seq2seq_model(session=sess,
                                     hparams=hparams,
                                     forward_only=True)
        # Unlike evaluation process, we just get output from a single sentence
        # each time, therefore, the batch_size here is 1
        model.batch_size = 1

        # We initialize vocab here, to load the encoder_inputs vocab to
        # encoder the input sentence from user
        # While the rev_vocab is the reverse vocab of decoder vocab to
        # decode the logits output from model into human sentence
        src_vocab, _ = data_utils.initialize_vocab("vocab.in")
        _, rev_vocab = data_utils.initialize_vocab("vocab.ou")

        print("Welcome COMP7404! I'm a chatbot! Say something!")
        while True:
            print("> ", end='')
            sys.stdout.flush()
            sentence = sys.stdin.readline()

            if len(sentence) > 0 and sentence[-1] == '\n':
                sentence = sentence[:-1]
            # Type exit to exit the robot
            if sentence == 'exit':
                break
            response = get_predicted_sentence(session=sess,
                                              input_sentence=sentence,
                                              src_vocab=src_vocab,
                                              rev_vocab=rev_vocab,
                                              seq2seq_model=model)
            print(response)


def chatbot():
    hparams = get_hparams()

    # The mode params can be triggered by terminal param
    if hparams.mode == 'train':
        train(hparams)
    elif hparams.mode == 'chat':
        chat(hparams)


if __name__ == '__main__':
    chatbot()
