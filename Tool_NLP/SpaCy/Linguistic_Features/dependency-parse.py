# -*- coding: utf-8 -*-

"""
    Software: Dependency Parse
    Tutorial: <https://spacy.io/usage/linguistic-features#section-dependency-parse>
    Description: Analisador de dependência sintática.
    Date: 21/06/2018
    Author: RodriguesFAS
    Contact: <franciscosouzaacer@gmail.com>
"""

import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp(u"Autonomous cars shift insurance liability toward manufacturers")
for chunk in doc.noun_chunks:
    print(chunk.text, chunk.root.text, chunk.root.dep_,
          chunk.root.head.text)