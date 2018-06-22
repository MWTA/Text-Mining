# -*- coding: utf-8 -*-

"""
    Software: Test 01
    Description:
    Date: 20/06/2018
    Author: Rodriguesfas
    Contact: <franciscosouzaacer@gmail.com>
"""

import spacy
from nltk.corpus import wordnet

nlp = spacy.load('en')


path_root = '/home/rodriguesfas/Workspace/Text-Mining/Lexical_Semantics/Experiment/data/wesendit_wfgL66Ct56WR/SemEval2018Task7-testing/phrase_test.txt'


def load_data(path_dataset):
    temp = []
    with open(path_dataset) as file:
        for line in file:
            temp.append(unicode(line.strip().decode('utf8')))
    return temp


def pos_tagging(doc):
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
              token.shape_, token.is_alpha, token.is_stop)


def main():
    dataset = load_data(path_root)

    for line in dataset:
        doc = nlp(line)
        pos_tagging(doc)


if __name__ == '__main__':
    main()
