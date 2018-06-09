    #!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
    Software: Naive Bayes
    Description: Algoritimo Naive Bayes aplicado ao dataset polarity.
    Date: 14/05/2018
    Author: RodrigueFAS 
'''


import os
import re
import sys
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

reload(sys)
sys.setdefaultencoding('utf-8')

dir_root = '/home/rodriguesfas/Workspace/Mining-Text/MT_2018_PPGIA/AULA_05_06/LE_04/review_polarity/txt_sentoken/'
dir_neg = 'neg/'
dir_pos = 'pos/'
dir_out = 'out/'

files_pos = dir_root + dir_pos
files_neg = dir_root + dir_neg


def remove_numbers(data):
    return re.sub('[-|0-9]', ' ', data)


def remove_special_characters(data):
    return re.sub(r'[-_./?,`":;=+()|@#$%&*^~\']', '', data)


def remove_stopwords(data):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    return [i for i in data.split() if not i in stopwords]


def stemmer(data):
    stemmer = nltk.stem.RSLPStemmer()
    words = []

    for word in data.split():
      words.append(stemmer.stem(word))

    return (" ".join(words))

# Load dataset and pre-processing.
def loadData(work_dir, label):
    dataset = open(dir_root + dir_out + 'pol.data.csv', 'a+')

    # navegation files
    for file in os.listdir(work_dir):
        if file.endswith('.txt'):
            # read file to lowercase text.
            words_file = open(work_dir + file).read().lower()

            # remove numbers
            words_file = remove_numbers(words_file)

            # Remove caracter special.
            words_file = remove_special_characters(words_file)

            # Remove stop-words
            words_file = remove_stopwords(words_file)

            # Adiciona label aos dados.
            words_file = words_file, label

            words_file = remove_special_characters(str(words_file))

            words_file = str(words_file).replace("'", '').replace('(', '').replace('[', '').replace(
                ']', '').replace(',', '').replace(')', '').replace('0', ', 0').replace('1', ', 1').replace('x,', '').replace('\\', '')

            words_file = stemmer(words_file)

            dataset.write(words_file + '\n')

    dataset.close()


# Run
print(">> Processando diretório (neg)... ")
loadData(files_neg, 0)

print(">> Processando diretório (pos)... ")
loadData(files_pos, 1)

print(">> Dataset gerado!")
