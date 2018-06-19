#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Description: Problema 03 - Sumarização.
    Date: 17/05/2018
    Author: RodriguesFAS
    Email: franciscosouzaacer@gmail.com
    License: MIT

    Tutorial
        <https://dev.to/davidisrawi/build-a-quick-summarizer-with-python-and-nltk>
"""

import os
import sys
import nltk

reload(sys)
sys.setdefaultencoding('utf-8')


path = '/home/rodriguesfas/Workspace/Mining-Text/MT_2018_PPGIA/AULA_07/LE_05/News/'


def load_corpus(news):
    return open(news).read().lower()


def pre_processing(news):
    words_tolower = news.lower()
    words_tokens = nltk.word_tokenize(words_tolower)
    words_sent = nltk.sent_tokenize(words_tokens)
    return words_sent


def main():
    with open(path + "news1.txt") as file:
        for line in file:
            if line != "\n":
                print(pre_processing(line))


# Run
main()
