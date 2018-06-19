#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Software: Pre-Processing-01
    
    Description: 
        - Processa todos arquivos, dos diretórios neg_ e pos_ do dataset que contem os registros
         já separados em arquivos .TXT;

        - Converte todos para caracteres minusculos;

        - Remove caracteres numéricos;

        - Remove caracteres especiais;

        - Remove stopwords usando uma lista de stopwords;

        - Stemmer reduz a raiz das palavras;

        - Remove palavras que contenha menos de 4 catacteres.

        - Adicionar rótulos: "1" positivos ou "0" negativo para cada arquivo.
        - Remove lixo gerado;

        - Juntar todos os dados em um único arquivo. Sendo assim, um novo dataser é gerado com o 
         dados para treinamento contendo n linhas "documentos" e 2 colunas, sendo que na 1ª coluna
         as palavras e na 2ª coluna o rótulo;

        - Pega o dataset de treinamento gerado e com base nele gera um novo dataset para teste 
          "sem rótulos". 
    
    Register: 21/05/2018
    
    Authors: 
        RodriguesFAS; 
        "coloquem  nome de vocês :)"
    
    Email: 
        <franciscosouzaacer@gmail.com> 
"""

import re
import os
import csv
import sys
import nltk


reload(sys)
sys.setdefaultencoding('utf-8')


path_root = '/home/rodriguesfas/Workspace/k-nn/data/'
path_dataset = 'polarity-detection-200-reviews/'

path_neg = 'neg_100/'
path_pos = 'pos_100/'
path_out = 'out/'

file_stopwords = 'stopwords/stopwords.txt'

path_list_stopwords = path_root + file_stopwords

files_neg = path_root + path_dataset + path_neg
files_pos = path_root + path_dataset + path_pos

path_train = path_root + path_out + 'generated-polarity-train.csv'
path_test = path_root + path_out + 'generated-polarity-test.csv'


label_pos = '1'
label_neg = '0'


TEST = True


def LOG(text):
    if TEST is True:
        print(">> " + text)


def remove_numbers(document):
    LOG('Remove numbers..')
    return re.sub('[-|0-9]', ' ', document).strip()


def remove_special_characters(document):
    LOG('Remove characters special..')
    document = re.sub(r'[-_./?!,`":;=+()<>|@#$%&*^~\']', '', document)
    document = "".join(document.splitlines())
    document = ' '.join(document.split())
    return document.strip()


def clean_document(document):
    LOG('Cleaning content..')

    document = document.replace('(', '').replace(')', '').replace("'", '').replace(
        ',', '').replace('\\n', '').replace('0', ', 0').replace('1', ', 1').replace('[', '').replace(']', '')

    return document


def remove_very_small_words(document):
    document = re.sub(r'\b\w{1,3}\b', '', document)
    return document.strip()


def remove_stopwords(document):
    LOG('Removing stopwords..')

    list_stopwords = open(path_list_stopwords).read()
    words_filtered = document[:]
    words_filtered = [i for i in document.split() if not i in list_stopwords]

    return (" ".join(words_filtered))


def stemmer(document):
    LOG('Stemming..')

    stemmer = nltk.stem.RSLPStemmer()
    words = []

    for word in document.split():
        words.append(stemmer.stem(word))

    return (" ".join(words))


def load_document(path, label):
    dataset_train = open(path_train, 'a+')

    for file in os.listdir(path):
        LOG('Processing data file: ' + file)

        if file.endswith('.txt'):
            document = open(path + file).read().lower()

            document = remove_numbers(document)
            document = remove_special_characters(str(document))
            document = remove_stopwords(document)
            document = stemmer(document)
            document = remove_very_small_words(str(document))

            LOG('Labeling content..')
            document = str(document), label
            document = clean_document(str(document))

            dataset_train.write(document + '\n')
    dataset_train.close()
    LOG('Dataset Train generated!')


def generater_test(path):
    LOG('Generating dataset from test..')
    dataset_test = open(path_test, 'a+')
    with open(path) as documents:
        for document in documents:
            document = document.replace(', 0', '').replace(', 1', '')
            dataset_test.write(document)
    dataset_test.close()
    LOG('Dataset Test generated!')


def main():
    LOG('Started!')

    LOG('Processing directory neg_ ...')
    load_document(files_neg, label_neg)

    LOG('Processing directory pos_ ...')
    load_document(files_pos, label_pos)

    generater_test(path_train)

    LOG('Finalized!')



# Run
if __name__ == '__main__':
    main()
