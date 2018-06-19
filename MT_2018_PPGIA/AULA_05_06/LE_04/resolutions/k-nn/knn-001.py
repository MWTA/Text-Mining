#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Software: K-NN
    Description:
    Date: 23/05/2018
    Authors: 
        RodriguesFAS; 
        "coloquem  nome de vocês :)"
    Email: 
        <franciscosouzaacer@gmail.com> 
"""

import re
import sys
import nltk
import random
import pandas as pd

from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


reload(sys)
sys.setdefaultencoding('utf-8')

path_root = '/home/rodriguesfas/Workspace/k-nn/data/out/'

path_data_train = path_root + 'generated-polarity-train.csv'
path_data_test = path_root + 'generated-polarity-test.csv'
path_data_predictions = path_root + 'generated-polarity-predictions.csv'

TEST = True

doc_id_lable = {}


def LOG(text):
    if TEST is True:
        print(">> " + text)


def bag_of_words():
    return "null"


def training(path):
    LOG('Training..')
    data = []
    index = 0
    col = ['document', 'label']

    with open(path) as documents:
        for document in documents:
            temp = [document.split(",")[0], document.split(",")[1]]
            data.append(temp)
            doc_id_lable[index] = str(document.split(",")[1]).strip()
            index += 1

    df = pd.DataFrame(data, columns=col)
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(df['document'].values)
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(X)

    # print df

    return tfidf


def test(path):
    LOG('Testing..')
    data = []

    with open(path) as documents:
        for document in documents:
            temp = document.split(",")[0]
            data.append(temp)

    df = pd.DataFrame(data, columns=['document'])
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(df['document'].values)
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(X)

    return tfidf


def cosine_similarity(d1, d2):
    return 1 - spatial.distance.cosine(d1, d2)


def knn(train_vector, test_vector, k):
    all_distances = {}
    for t in range(len(train_vector)):
        dist = cosine_similarity(train_vector[t], test_vector)
        all_distances[t] = dist
    return [(k, all_distances[k]) for k in sorted(all_distances, key=lambda x:all_distances[x], reverse=True)][:k]


def get_predicted_label(nearest):
    pos = 0
    neg = 0
    dict = {k: v for k, v in nearest}

    for k in dict:
        if doc_id_lable[k] == '1':
            pos += 1
        else:
            neg += 1

    if (pos == neg):
        gen_lab = random.sample(set([0, 1]), 1)
    else:
        gen_lab = '1' if pos > neg else '0'
        
    return str(gen_lab).replace('[', '').replace(']', '')


def get_accuracy(testSet, dataset_predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == dataset_predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


def main():
    dataset_predictions = []
    gen_labels = []
    k = 5

    LOG('Started!')

    doc_training = training(path_data_train)
    doc_test = test(path_data_train)

    doc_training_array = doc_training.toarray()
    doc_test_array = doc_test.toarray()

    doc_result = open(path_data_predictions, 'w')
    index = 0
    for document in doc_test_array:
        nearest = knn(doc_training_array, document, k)
        lab = get_predicted_label(nearest)
        doc_result.write(lab + "\n")
        gen_labels.append(lab)
        dataset_predictions.append(lab)
        print "Calculated Label: " + lab
        index += 1
    doc_result.close()

    accuracy = get_accuracy(doc_id_lable, dataset_predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

    LOG('Finalized!')


if __name__ == '__main__':
    main()
