#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Software: Naive Bayes
    Description: Algoritimo Naive Bayes aplicado ao dataset polarity.
    Date: 14/05/2018
    Author: RodrigueFAS 
'''


import io
import os
import re
import math
import nltk
import random

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


dir_root = '/home/rodriguesfas/Workspace/Mining-Text/MT_2018_PPGIA/AULA_05_06/LE_04/review_polarity/txt_sentoken/'
dir_neg = 'neg/'
dir_pos = 'pos/'

files_pos = dir_root + dir_pos
files_neg = dir_root + dir_neg


# Load dataset and pre-processing.
def loadData(work_dir, file):
	words_file = open(work_dir + file).read().lower()

	# Remove number.
	words_file = re.sub('[-|0-9]', '', words_file)

	# Remove caracter special.
	words_file = re.sub(r'[-./?,*":;()\']', '', words_file)
	words_file = words_file.replace("\n", '').replace(
		"' ,'", ''). replace("', '", '')

	return words_file


def remove_stopwords(work_dir):
	stop_words = set(nltk.corpus.stopwords.words('english'))
	tokens_words = nltk.word_tokenize(loadData(work_dir))
	filtered_sentence = [word for word in tokens_words if not word in stop_words]
	filtered_sentence = []
	for word in tokens_words:
		if word not in stop_words:
			filtered_sentence.append(word)
	return filtered_sentence


# Divide o dataset em conjuntos de treinamento e de teste, aleatoriamente.
def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)

	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]


# Main
def main(work_dir):

	# Proporção da divisão do dataset
	# 67% treino e 33% teste, segundo a literatura.
	splitRatio = 0.67

	train, test = splitDataset(os.listdir(work_dir), splitRatio)
	print(' Dividido aleatoriamente {0} arquivos \n Treinamento: {1} \n Teste: {2}').format(
		len(os.listdir(work_dir)), len(train), len(test))

	# prepare o model
	for file in os.listdir(work_dir):
		if file.endswith('.txt'):
			words = loadData(work_dir, file)

	# Test model


# Run
main(files_pos)
