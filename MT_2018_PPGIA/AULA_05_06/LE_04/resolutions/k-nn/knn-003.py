#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Software: Knn

    Description: Classifica textos binária.
        
        Passos

        - 1º Carrega dataset.
        - 2º Cria um DataFrame que recebe o dataset carregado e adiciona nomes as colunas 
            da tabela criada.
        - 3º Remove stopwords com base no dicionário do LNTK.
        - 4º Remove stopwords com base na stoplist propria.
        - 5º Reduz as palavras a raiz com Stemming.
        - 6º Remove palavras muito pequenas "que contem menos que 4 caracteres".
        - 7º Embaralha as linhas da tabela, para os dados não ficarem sequênciais.
        - º 
        - º 
    
    Date: 31/05/2018
"""

import os
import re
import sys
import nltk
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import array
from scipy import spatial
from nltk.corpus import stopwords

# scikit-learn k-fold cross-validation
from sklearn.model_selection import KFold

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


reload(sys)
sys.setdefaultencoding('utf-8')

path_root = '/home/rodriguesfas/Workspace/k-nn/data/out/'

path_dataset_input = path_root + 'generated-polarity-data.csv'
path_dataset_predictions = path_root + 'generated-polarity-predictions-data.csv'

path_stop_list = path_root + '../stopwords/stopwords.txt'


"""
    Debug Console
    Flag
        True 
        False
"""
TEST = True

k = 2
split = 5
doc_id_lable = {}
label_predictions = {}
accuracy = []


def LOG(text):
    if TEST is True:
        print(">> " + text)


def remove_stopword_nltk(dataset):
    cachedStopWords = stopwords.words("english")
    dataset = ' '.join(
        [word for word in dataset.lower().split() if word not in cachedStopWords])
    return dataset.strip().strip()


def remove_stopword_list(dataset):
    stop_list = open(path_stop_list).read()
    words_filtered = dataset[:]
    words_filtered = [i for i in dataset.split() if not i in stop_list]
    return (" ".join(words_filtered)).strip()


def stemmer(dataset):
    stemmer = nltk.stem.RSLPStemmer()
    words = []
    for word in dataset.split():
        words.append(stemmer.stem(word))
    return (" ".join(words)).strip()


def remove_very_small_words(dataset):
    dataset = re.sub(r'\b\w{1,3}\b', '', dataset)
    return dataset.strip()


def load_dataset(path_dataset):
    LOG('Loading data..')
    with open(path_dataset) as documents:
        data = []
        index = 0
        for document in documents:
            temp = [document.split(",")[0], document.split(",")[1].strip()]
            data.append(temp)
            doc_id_lable[index] = str(document.split(",")[1]).strip()
            index += 1
    df = pd.DataFrame(data)
    df.columns = ['document', 'label']

    LOG('Pre-processing of data..')
    LOG('Removing stopwords nltk..')
    df['document'] = df['document'].apply(lambda x: remove_stopword_nltk(x))
    LOG('Removing stoplist..')
    df['document'] = df['document'].apply(lambda x: remove_stopword_list(x))
    LOG('Stemming..')
    # df['document'] = df['document'].apply(lambda x: stemmer(x))
    LOG('Removing too short words..')
    df['document'] = df['document'].apply(lambda x: remove_very_small_words(x))
    LOG('Shuffling Positive and Negative Data..')
    df = df.take(np.random.permutation(len(df)))
    return df


def split_dateset(dataset, split):
    files = array(
        [
            'split1.csv',
            'split2.csv',
            'split3.csv',
            'split4.csv',
            'split5.csv',
        ]
    )

    k_fold = KFold(n_splits=split, shuffle=True, random_state=1)

    for train, test in k_fold.split(files):
        print('train: %s, test: %s' % (files[train], files[test]))

    return "null"


def training(dataset):
    LOG('Training..')
    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(dataset['document'].values)
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(X)
    return tfidf


def cosine_similarity(d1, d2):
    return 1 - spatial.distance.cosine(d1, d2)


def knn(train_vector, test_vector, k):
    all_distances = {}
    for index in range(len(train_vector)):
        dist = cosine_similarity(train_vector[index], test_vector)
        all_distances[index] = dist
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


def calcular_accuracy(testSet, dataset_predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == dataset_predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


def graph_accuracy():

    x = [1, 2, 3, 4, 5, 6, 7, 8] 

    # plotting the points
    plt.plot(x, accuracy, color='green', linestyle='dashed', linewidth=3,
             marker='o', markerfacecolor='blue', markersize=12)

    # setting x and y axis range
    plt.ylim(1, 8)
    plt.xlim(1, 8)
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Graph Accuracy')
    plt.show()


def main():
    LOG('Started!')

    dataset = load_dataset(path_dataset_input)

    # split_dateset(dataset, split)

    tfidf_train = training(dataset)
    train_array = tfidf_train.toarray()

    tfidf_test = training(dataset)
    test_array = tfidf_test.toarray()

    for j in range(2, 10):
        dataset_predictions = open(path_dataset_predictions, 'w')
        index = 0
        for document in test_array:
            nearest = knn(train_array, document, k=+1)
            label = get_predicted_label(nearest)
            dataset_predictions.write(label + "\n")
            label_predictions[index] = label
            print index, "Calculated Label: " + label
            index += 1
        dataset_predictions.close()

        ac = calcular_accuracy(doc_id_lable, label_predictions)
        accuracy.append(ac)
        print('Accuracy: ' + repr(ac) + '%')

    graph_accuracy()
    print accuracy
    LOG('Finalized!')


if __name__ == '__main__':
    main()
