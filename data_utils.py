from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import progressbar
import re
import operator

DATA_PATH = 'data/'
MOVIE_LINES_FILE = 'movie_lines.txt'
MOVIE_CONVOS_FILE = 'movie_conversations.txt'
VOCAB_COUNT_THRESHOLD = 2


def get_lineid_content():
    lineid_content = {}
    lines_file_path = os.path.join(DATA_PATH + MOVIE_LINES_FILE)

    with open(lines_file_path, 'r', errors='ignore') as f:
        # +++$+++ is used to split the section in a single line
        # A correct formed line includes five sections
        # The first section is lineID
        # The last section is line content

        for line in f:
            line_sections = line.split(' +++$+++ ')
            assert len(line_sections) == 5
            if line_sections[4][-1] == '\n':
                line_sections[4] = line_sections[4][:-1]
            lineid_content[line_sections[0]] = line_sections[4]

    return lineid_content


def get_convos():
    convos = []
    convos_file_path = os.path.join(DATA_PATH, MOVIE_CONVOS_FILE)

    with open(convos_file_path, 'r', errors='ignore') as f:
        # +++$+++ is used to split the section in a single line
        # A correct formed line includes four sections
        # The last section is lineIDs in each conversation

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

            input_length = len(lineid_content[each_convo[index]].split(' '))
            output_length = len(lineid_content[each_convo[index]].split(' '))

            if input_length < 50 and output_length < 50 and \
                    output_length < 2 * input_length:
                input_set.append(lineid_content[each_convo[index]])
                output_set.append(lineid_content[each_convo[index + 1]])
            else:
                continue

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
        f.write('<unk>' + '\n')
        f.write('<s>' + '\n')
        f.write('</s>' + '\n')
        vocab_count = 3
        for word in vocab_sorted:
            # Only record the words over particular count threshold
            # Otherwise we will treat the rare words as <unk>
            if vocab[word[0]] < VOCAB_COUNT_THRESHOLD:
                break
            else:
                f.write(word[0] + '\n')
            vocab_count += 1


def data_preprocessing():
    lineid_content = get_lineid_content()
    print('Read movie_lines.txt file complete...')
    convos = get_convos()
    print('Read movie_conversations.txt file complete...')
    print('Building train, dev, test set...')
    get_train_dev_test_set(lineid_content, convos)
    print('Building vocabulary file...')
    construct_vocab_file('train.in')
    construct_vocab_file('train.ou')


if __name__ == '__main__':
    data_preprocessing()
