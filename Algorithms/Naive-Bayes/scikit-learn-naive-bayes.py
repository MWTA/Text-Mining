#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Software: Naive Bayes
    Description: Algoritimo Naive Bayes aplicado ao dataset polarity.
    Date: 15/05/2018
    Author: RodrigueFAS 

    Tutorial: <http://scikit-learn.org/stable/modules/naive_bayes.html>
'''


import numpy as np

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()

gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)


print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0], (iris.target != y_pred).sum()))
