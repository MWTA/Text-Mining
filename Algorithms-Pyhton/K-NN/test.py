#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Software: Pre-Processing
    Description: Pré procesamento dos dados, limpesa dos dados, separação dos dados.
    Date: 21/05/2018
    Authors: 
        RodriguesFAS; 
        "coloquem  nome de vocês :)"
    Email: 
        <franciscosouzaacer@gmail.com> 
"""


import os
import re
import csv
import sys
import nltk
import random


reload(sys)
sys.setdefaultencoding('utf-8')


dir_root = '/home/rodriguesfas/Workspace/k-nn/data/'
dir_dataset = 'polarity-detection-200-reviews/'
dir_stopwords = 'stopwords/stopwords.txt'
dir_neg = 'neg_100/'
dir_pos = 'pos_100/'
dir_out = 'out/'

files_neg = dir_root + dir_dataset + dir_neg
files_pos = dir_root + dir_dataset + dir_pos
files_list_stopwords = dir_root + dir_stopwords

file_corpus = 'pol.data.csv'
file_training = 'training.data.csv'
file_test = 'test.data.csv'

training_set = []
test_set = []
split = 0.80

TEST = True


def LOG(text):
    if TEST is True:
        print(">> " + text)


def remove_numbers(data):
    LOG('Remove numbers..')
    return re.sub('[-|0-9]', ' ', data)


def remove_special_characters(data):
    LOG('Remove characters special..')
    return re.sub(r'[-_./?,`":;=+()|@#$%&*^~\']', '', data)


def remove_stopwords(data):
    LOG('Remove stopwords..')

    list_stopwords = open(files_list_stopwords).read()
    words_filtered = data[:]
    words_filtered = [i for i in data.split() if not i in list_stopwords]

    return (" ".join(words_filtered))


def stemmer(data):
    stemmer = nltk.stem.RSLPStemmer()
    words = []

    for word in data.split():
        words.append(stemmer.stem(word))

    return (" ".join(words))


def load_corpus(work_dir, label):
    dataset = open(dir_root + dir_out + 'pol.data.csv', 'a+')

    for file in os.listdir(work_dir):
        LOG('Processing file: ' + file)
        if file.endswith('.txt'):
            words_file = open(work_dir + file).read().lower()
            words_file = remove_numbers(words_file)
            words_file = remove_special_characters(words_file)
            words_file = remove_stopwords(words_file)
            words_file = words_file, label
            words_file = remove_special_characters(str(words_file))

            words_file = str(words_file).replace("'", '').replace('(', '').replace('[', '').replace(
                ']', '').replace(')', '').replace('0', ', 0').replace('1', ', 1').replace('x,', '').replace('\\', '')

            words_file = stemmer(words_file)
            dataset.write(words_file + '\n')
    dataset.close()


def separate_data(path, split, training_set=[], test_set=[]):
    LOG("Processing data...")

    with open(path, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(2):
                dataset[x][y] = dataset[x][y]
            if random.random() < split:
                training_set.append(str(dataset[x]))
            else:
                test_set.append(str(dataset[x]))

    LOG('Separad data: ')
    LOG("Train: " + repr(len(training_set)))
    LOG("Test: " + repr(len(test_set)))


def save_data():
    LOG('Salve datas train e test...')

    # training
    file_training_set = open(dir_root + dir_out + file_training, 'w')    
    file_training_set.writelines(str(training_set) + '\n')
    file_training_set.close()

    # test
    file_test_set = open(dir_root + dir_out + file_test, 'w')
    file_test_set.writelines(str(test_set) + '\n')
    file_test_set.close()


def main():
    LOG('Processing directory (neg)...')
    load_corpus(files_neg, 0)

    LOG('Processing directory (pos)...')
    load_corpus(files_pos, 1)

    separate_data(dir_root + dir_out + file_corpus, split, training_set, test_set)

    save_data()

    LOG('Finalized!')


# Run
main()
