# -*- coding: utf-8 -*-

"""
    Software: Test 01
    Description:
    Date: 20/06/2018
    Author: Rodriguesfas
    Contact: <franciscosouzaacer@gmail.com>
"""

import spacy

nlp = spacy.load('en')


path_root = '/home/rodriguesfas/Workspace/Text-Mining/Lexical_Semantics/Experiment/data/wesendit_wfgL66Ct56WR/SemEval2018Task7-testing/phrase_test.txt'


def load_data(path_dataset):
    temp = []
    with open(path_dataset) as file:
        for line in file:
            temp.append(unicode(line.strip().decode('utf8')))
    return temp


def main():
    data = load_data(path_root)
    for line in data:
        doc = nlp(line)
        for token in doc:
            print(token.text, token.pos_, token.dep_)


if __name__ == '__main__':
    main()
