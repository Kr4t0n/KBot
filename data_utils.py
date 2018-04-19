# This data_utils.py file is modified from Google Tensorflow
# old version of model of rnn.translate.seq2seq_model
# You can find the original file from Google's Tensorflow
# Github repository history
#
# This file has added some useful utility functions to
# the needs of preprocessing data, you can tweet the data
# utility functions with your own needs

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import progressbar
import re
import operator

DATA_PATH = 'data/'
CPT_DIR = 'checkpoints/'
MOVIE_LINES_FILE = 'movie_lines.txt'
MOVIE_CONVOS_FILE = 'movie_conversations.txt'
VOCAB_COUNT_THRESHOLD = 2

PAD_ID = 0
UNK_ID = 1
START_ID = 2
END_ID = 3


BUCKETS = [(5, 10), (10, 15), (20, 25), (40, 50)]

# The following functions are used to retrieve the
# Cornell Movie-Dialogs Corpus dataset


def get_lineid_content():
    """ Get the pair of lineID and content
        movie_lines.txt file includes the corresponding
        conversation content to each lineID
    """
    lineid_content = {}
    lines_file_path = os.path.join(DATA_PATH + MOVIE_LINES_FILE)

    with open(lines_file_path, 'r', errors='ignore') as f:
        # +++$+++ is used to split the section in a single line
        # A correct formed line includes five sections
        # The first section is lineID
        # The last section is line content
        # Here we only need lineID and content

        for line in f:
            line_sections = line.split(' +++$+++ ')
            assert len(line_sections) == 5
            if line_sections[4][-1] == '\n':
                line_sections[4] = line_sections[4][:-1]
            lineid_content[line_sections[0]] = line_sections[4]

    return lineid_content


def get_convos():
    """ Reconstruct the movie conversation into pair of dialog
        into list of ordered lineIDs
    """
    convos = []
    convos_file_path = os.path.join(DATA_PATH, MOVIE_CONVOS_FILE)

    with open(convos_file_path, 'r', errors='ignore') as f:
        # +++$+++ is used to split the section in a single line
        # A correct formed line includes four sections
        # The last section is list of lineIDs in each conversation

        for line in f:
            line_sections = line.split(' +++$+++ ')
            assert len(line_sections) == 4
            convos.append(line_sections[3][1:-2].replace('\'', '').split(', '))

    return convos


def get_train_dev_test_set(lineid_content, convos):
    # Construct the corresponding input and output conversation pair
    # Turn lineID into content

    input_set, output_set = [], []
    for each_convo in convos:
        for index, lineid in enumerate(each_convo[:-1]):
            # Input and output data should be roughly less than 50 words
            # And we hope that output length is less than 2 times of input
            # Cause when output length is bigger than 2 times of input
            # There's highly chance that it makes no sense to infer the output

            input_length = len(
                lineid_content[each_convo[index]].split(' '))
            output_length = len(
                lineid_content[each_convo[index + 1]].split(' '))

            if input_length < 50 and output_length < 50 and \
                    output_length < 2 * input_length:
                input_set.append(lineid_content[each_convo[index]])
                output_set.append(lineid_content[each_convo[index + 1]])
            else:
                continue

    # Make sure than the length of input_set is equal to output_set
    # Since every input sentence corresponds to one output response
    assert len(input_set) == len(output_set)

    # Split the pair into train set and test set
    # Train set : Dev set : Test set = 7 : 2 : 1
    filenames = ['train.in', 'train.ou',
                 'dev.in', 'dev.ou', 'test.in', 'test.ou']
    file_pool = []
    for each_file in filenames:
        file_pool.append(open(os.path.join(DATA_PATH, each_file), 'w+'))

    # Random choose 30% data into non train set
    non_train_set_index = random.sample(
        [i for i in range(len(input_set))], len(input_set) * 3 // 10)
    dev_set_index = random.sample(
        non_train_set_index, len(non_train_set_index) // 3)

    # Split the data set
    for i in progressbar.progressbar(range(len(input_set))):
        if i not in non_train_set_index:
            # i is train_set_index
            file_pool[0].write(input_set[i] + '\n')
            file_pool[1].write(output_set[i] + '\n')
        elif i in dev_set_index:
            # i is dev_set_index
            file_pool[2].write(input_set[i] + '\n')
            file_pool[3].write(output_set[i] + '\n')
        else:
            # i is test_set_index
            file_pool[4].write(input_set[i] + '\n')
            file_pool[5].write(output_set[i] + '\n')

    for file in file_pool:
        file.close()


def basic_tokenizer(line, normalize_digits=False):
    # Split a single line into words with tokenizer
    # This basic tokenizer is provided by Google's translate project
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT_RE = re.compile("([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT_RE, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            words.append(token)
    return words


def construct_vocab_file(filename):
    # Construct the vocabulary file
    input_file = os.path.join(DATA_PATH, filename)
    output_file = os.path.join(DATA_PATH, 'vocab.{}'.format(filename[-2:]))
    vocab = {}

    with open(input_file, 'r') as f:
        for line in f.readlines():
            for word in basic_tokenizer(line, True):
                if word not in vocab:
                    vocab[word] = 0
                vocab[word] += 1

    vocab_sorted = sorted(
        vocab.items(), key=operator.itemgetter(1), reverse=True)
    with open(output_file, 'w+') as f:
        f.write('<pad>' + '\n')
        f.write('<unk>' + '\n')
        f.write('<s>' + '\n')
        f.write('</s>' + '\n')
        for word in vocab_sorted:
            # Only record the words over particular count threshold
            # Otherwise we will treat the rare words as <unk>
            if vocab[word[0]] < VOCAB_COUNT_THRESHOLD:
                break
            else:
                f.write(word[0] + '\n')

# The following functions are modified from Google's translate
# project of data_utils.py file


def initialize_vocab(vocab_path):
    """ Initialize for reading vocab
        Return the dictionary of vacab to wordID and
        a reverse vocab for decoding output
    """
    rev_vocab = []

    with open(os.path.join(DATA_PATH + vocab_path), 'r') as f:
        rev_vocab.extend(f.readlines())

    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])

    return vocab, rev_vocab


def sentence_to_token_ids(sentence, vocab):
    """ Reconstruct a word-based sentence into a
        wordIDs-based sentence
    """
    return [vocab.get(token, UNK_ID) for token in basic_tokenizer(sentence)]


def data_to_token_ids(src_path,
                      tgt_path,
                      vocab_path,
                      type):
    """ Reconstruct a word-based sentences dataset into a
        wordID-based sentences dataset
        For each line in the source dataset, we simply apply
        sentence_to_token_ids to it
    """
    vocab, _ = initialize_vocab(vocab_path)
    with open(src_path, 'r') as src_file:
        with open(tgt_path, 'w+') as tgt_file:
            for line in src_file:
                # Add START_ID to output
                if type == 'output':
                    token_ids = [START_ID]
                else:
                    token_ids = []

                token_ids.extend(sentence_to_token_ids(line, vocab))

                # Add END_ID to output
                if type == 'output':
                    token_ids.append(END_ID)
                tgt_file.write(" ".join([str(token_id)
                                         for token_id in token_ids]) + "\n")


def load_data(encoder_path, decoder_path, max_size=None):
    """ Load data from dataset
        Reconstruct the dataset into len(data_sets) = len(BUCKETS)
        each element inside data_sets is pairs of [encoder, decoder]
        encoder and decoder are represented as wordIDs
    """
    encoder_file = open(os.path.join(DATA_PATH, encoder_path), 'r')
    decoder_file = open(os.path.join(DATA_PATH, decoder_path), 'r')
    encoder, decoder = encoder_file.readline(), decoder_file.readline()
    data_sets = [[] for _ in BUCKETS]
    size = 0

    while encoder and decoder and (not max_size or size < max_size):
        encoder_ids = [int(id) for id in encoder.split()]
        decoder_ids = [int(id) for id in decoder.split()]

        for bucket_id, (encoder_size, decoder_size) in enumerate(BUCKETS):
            # To insert encoder and decoder into the most proper size of bucket
            if len(encoder_ids) < encoder_size and \
                    len(decoder_ids) < decoder_size:
                data_sets[bucket_id].append([encoder_ids, decoder_ids])
                break
        encoder, decoder = encoder_file.readline(), decoder_file.readline()

    return data_sets


def get_vocab_size(vocab_path):
    """ Get the vocab size, contribute to dynamically
        add vocab_size into hyper parameters
    """
    with open(os.path.join(DATA_PATH, vocab_path), 'r') as f:
        vocab_size = sum(1 for _ in f)

    return vocab_size


def data_preprocessing():
    """ Data preprocessing procedure, mainly for process
        Cornell Movie-Dialogs Corpus
    """
    lineid_content = get_lineid_content()
    print('Read movie_lines.txt file complete...')
    convos = get_convos()
    print('Read movie_conversations.txt file complete...')
    print('Building train, dev, test set...')
    get_train_dev_test_set(lineid_content, convos)
    print('Building vocabulary file...')
    construct_vocab_file('train.in')
    construct_vocab_file('train.ou')


def data_tokenizing():
    """ Reconstruct train.in, dev.in, test.in
                    train.ou, dev.ou, test.ou
        into wordIDs based sentences
    """
    data_to_token_ids(src_path=os.path.join(DATA_PATH, 'train.in'),
                      tgt_path=os.path.join(DATA_PATH, 'train_ids.in'),
                      vocab_path=os.path.join(DATA_PATH, 'vocab.in'),
                      type='input')
    data_to_token_ids(src_path=os.path.join(DATA_PATH, 'dev.in'),
                      tgt_path=os.path.join(DATA_PATH, 'dev_ids.in'),
                      vocab_path=os.path.join(DATA_PATH, 'vocab.in'),
                      type='input')
    data_to_token_ids(src_path=os.path.join(DATA_PATH, 'test.in'),
                      tgt_path=os.path.join(DATA_PATH, 'test_ids.in'),
                      vocab_path=os.path.join(DATA_PATH, 'vocab.in'),
                      type='input')
    data_to_token_ids(src_path=os.path.join(DATA_PATH, 'train.ou'),
                      tgt_path=os.path.join(DATA_PATH, 'train_ids.ou'),
                      vocab_path=os.path.join(DATA_PATH, 'vocab.ou'),
                      type='output')
    data_to_token_ids(src_path=os.path.join(DATA_PATH, 'dev.ou'),
                      tgt_path=os.path.join(DATA_PATH, 'dev_ids.ou'),
                      vocab_path=os.path.join(DATA_PATH, 'vocab.ou'),
                      type='output')
    data_to_token_ids(src_path=os.path.join(DATA_PATH, 'test.ou'),
                      tgt_path=os.path.join(DATA_PATH, 'test_ids.ou'),
                      vocab_path=os.path.join(DATA_PATH, 'vocab.ou'),
                      type='output')


if __name__ == '__main__':
    data_preprocessing()
    data_tokenizing()
