#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Software: K-NN

    Description:
        - Carrega dados, usando DataFrame Pandas.
        - Embaralha as linhas do DataFrame.
        - Separa os dados para testes e treinamento.
        - Calcula TF-IDF dos dados "Teinamento".
        - Calcula o vizinho mais próximo knn.
        - 

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
import numpy as np
import pandas as pd

from scipy import spatial

# scikit-learn k-fold cross-validation
from sklearn.model_selection import KFold

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


reload(sys)
sys.setdefaultencoding('utf-8')

path_root = '/home/rodriguesfas/Workspace/k-nn/data/out/'

path_data = path_root + 'generated-polarity-train.csv'
path_data_result = path_root + 'generated-polarity-predictions.csv'

TEST = True

doc_id_lable = {}


def LOG(text):
    if TEST is True:
        print(">> " + text)


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


def cosine_similarity(d1, d2):
    return 1 - spatial.distance.cosine(d1, d2)


def knn(train_vector, test_vector, k):
    all_distances = {}
    for index in range(len(train_vector)):
        dist = cosine_similarity(train_vector[index], test_vector)
        all_distances[index] = dist
    return [(k, all_distances[k]) for k in sorted(all_distances, key=lambda x:all_distances[x], reverse=True)][:k]


def training(data_frame):
    LOG('Training..')

    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(data_frame['document'].values)
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(X)

    return tfidf


def load_dataset(path):
    LOG('Loading data..')
    data = []

    with open(path) as documents:
        index = 0
        for document in documents:
            temp = [document.split(",")[0], document.split(",")[1].strip()]
            data.append(temp)
            doc_id_lable[index] = str(document.split(",")[1]).strip()
            index += 1

    data_frame = pd.DataFrame(data)
    data_frame.columns = ['document', 'label']

    # altera posições das linhas da tabela aleatoriamente.
    data_frame_radom = data_frame.take(np.random.permutation(len(data_frame)))

    return data_frame_radom


def split_dataset(dataset, split):
    # data = np([0.1, 0.2, 0.3, 0.4, 0.5])

    # prepare cross validation
    k_fold = KFold(n_splits=split, shuffle=True, random_state=1)

    # enumerate splits
    # for train, test in k_fold.split(dataset):
    #    print('train: %s, test: %s' % (dataset.iloc[train[0]], dataset.iloc[test[1]]))

    # added some parameters
    result = next(k_fold.split(dataset), None)
    #print result

    dataset_train = dataset.iloc[result[0]]
    dataset_test = dataset.iloc[result[1]]

    return dataset_train, dataset_test


def main():
    LOG('Started!')

    gen_labels = []
    k = 3
    split = 5

    dataset = load_dataset(path_data)

    print dataset

    # separar dados.
    dataset_train, dataset_test = split_dataset(dataset, split)

    tfidf_train = training(dataset)
    train_array = tfidf_train.toarray()

    tfidf_test = training(dataset)
    test_array = tfidf_test.toarray()

    doc_result = open(path_data_result, 'w')
    index = 0
    for document in test_array:
        nearest = knn(train_array, document, k)
        lab = get_predicted_label(nearest)
        doc_result.write(lab + "\n")
        gen_labels.append(lab)
        print index, "Calculated Label: " + lab
        index += 1
    doc_result.close()

    LOG('Finalized!')


if __name__ == '__main__':
    main()
