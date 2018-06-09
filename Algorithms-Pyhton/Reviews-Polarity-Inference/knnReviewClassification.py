from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords
from scipy import spatial

import re
import random
import operator
import itertools
import numpy as np
import pandas as pd

cachedStopWords = stopwords.words("english")
columns = ['label', 'document']

path = '/home/rodriguesfas/Workspace/Text-Mining/Reviews-Polarity-Inference/data/'

# this function helps in preprocessing,
# By considreing only chars between a-z & A-z it also replaces the dot symbols and
# removesthe words that have length smaller than 4 chars


def replace_special_character(document):
    result = re.sub('[^a-zA-Z\n\.]', ' ', document).replace(".", "")
    result = ' '.join(result.split())
    result = "".join(result.splitlines())
    result = re.sub(r'\b\w{1,3}\b', '', result)
    return result.strip()

# this function also helps preprocessing but at thesecond stage by removing the stop words.
# Stop words are such ords that have very less significance in a sentence and are often repeated for ex : is, a, the etc


def removestopword(document):
    text = ' '.join([word for word in document.lower().split()
                     if word not in cachedStopWords])
    return text


# In this method we are reading the training data and creating a map which will have the documentNumber and the label found
# against that document in the training data, it also calls the prerocessing methods on each document in order to clean them
# After preprocesisng, we are keeping the data in a dataFrame from pandas. We calculate the termfrequency for each doument and then
# calculate the TfIDf : TermFrequency Inverse Document Frequency for each term and return the final result in a matrix.
# One point to note here is we have used max_features in our countVectorizer, we have done that inorder to maintain same
# dimension between the Training Data TfIdf and the Test Data TfIdf. Here we have randomly decided for a maximum of 2000 Features (cols)
# to be present in the matrix
docIdLable = {}


def readdocument():
    data = []
    files = [path + "train.dat"]

    for f in files:
        with open(f) as fl:
            idx = 0
            for lines in fl:
                temp = [lines[:2], lines[3:].strip(" ")]
                data.append(temp)
                docIdLable[idx] = lines[:2]
                idx += 1
    df = pd.DataFrame(data, columns=columns)
    df['document'] = df['document'].apply(lambda x: removestopword(x))
    df['document'] = df['document'].apply(lambda x: replace_special_character(x))
    vectorizer = CountVectorizer(min_df=1, max_features=2000)
    X = vectorizer.fit_transform(df['document'].values)
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(X)
    return tfidf

# This method is very similar to the one defined above, just that this method reads and preprocesses the testData. And thus
# it doesnt has to extract labels from each document, but as mentioned earlier in order to maintain same number of dimensions
# on both trainingData and testData we have used MaxFeatures to be 2000


def readTestdocument(location):
    data = []

    with open(location) as fl:
        for lines in fl:
            temp = lines.strip(" ")
            data.append(temp)
    df = pd.DataFrame(data, columns=['document'])
    df['document'] = df['document'].apply(lambda x: removestopword(x))
    df['document'] = df['document'].apply(lambda x: replace_special_character(x))
    vectorizer = CountVectorizer(min_df=1, max_features=2000)
    X = vectorizer.fit_transform(df['document'].values)
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(X)
    return tfidf


# the method to calculate CoSineSimilarity Between two vectors, this cosine similarity is then used to find the nearest neighbours
def cosineSimilarity(d1, d2):
    return 1 - spatial.distance.cosine(d1, d2)


# this Method calculates the nearest K neighbours of any vector from the training data set. It calls the cosineSimilarity subroutine to
# calculate the similarity between two vectors. After calculating similarity to all the vectors of the training data set, it sorts the dist in
# desencending order and selects the top k neighbours. And finally returns them as a map having DocumentId as key and CosineSimilarity as value
def kNearestNeighbours(trainVector, testVector, k):
    allDistances = {}
    for t in range(len(trainVector)):
        dist = cosineSimilarity(trainVector[t], testVector)
        allDistances[t] = dist
    return [(k, allDistances[k]) for k in sorted(allDistances, key=lambda x:allDistances[x], reverse=True)][:k]


# this is the method that generates the lables from the calculated neighbours label. If the neighbours are more positive we
#  assign positive label to this vector else if the neighbours aremove negative we assign negative label to it. Just in case,
# If both positive and negative neighbours are equal than we make a random decision about the label of current document.

def getPredictedLabel(nearDocs):
    pos = 0
    neg = 0
    dict = {k: v for k, v in nearDocs}
    for k in dict:
        print docIdLable[k]
        if docIdLable[k] == '+1':
            pos += 1
        else:
            neg += 1

    genLab = None
    if (pos == neg):
        genLab = random.sample(set([-1, 1], 1))
    else:
        genLab = '+1' if pos > neg else '-1'
    return genLab


# The main method of the program. We have used K = 5 in order to find the closest 5 neighbours of the vecor. Later after the label is
# assigned we write them to an external file. After all the labels have been generated we close the file.
if __name__ == '__main__':
    finalTrainTf = readdocument()
    print("train doc read completed")

    print docIdLable

    finalTestTf = readTestdocument(path + "test.dat")
    print("test doc read completed")

    finalTrainTfArr = finalTrainTf.toarray()
    finalTestTfArr = finalTestTf.toarray()

    genLabels = []
    k = 5
    f = open(path + 'result.dat', 'w')

    for tes in finalTestTfArr:
        nearest = kNearestNeighbours(finalTrainTfArr, tes, k)
        lab = getPredictedLabel(nearest)
        f.write(lab+"\n")
        genLabels.append(lab)
        print("calculated Label : ", lab)
    f.close()
