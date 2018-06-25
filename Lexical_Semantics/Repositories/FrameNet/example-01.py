"""
    Software: Example 01
    Description: FrameNet in NLTK
    Tutorial: <http://www.nltk.org/howto/framenet.html>
    Date: 23/06/2018
    Author: RodriguesFAS
    Contact: <franciscosouzaacer@gmail.com>
"""

import nltk
from pprint import pprint
from nltk.corpus import framenet as fn

# Obtem uma lista de todos os Frames no FrameNet.
print len(fn.frames())

# 
pprint(fn.frames(r'(?i)medical'))

# Obtem os detalhes de um Frame especifico.
f = fn.frame(256)
print f.ID
print f.name
print f.definition
print len(f.lexUnit)

pprint(sorted([x for x in f.FE]))
pprint(f.frameRelations)

#
fn.frames_by_lemma(r'(?i)a little')

# Visualiza Documento Anotados
docs = fn.documents()
print len(docs)
pprint(sorted(docs[0].keys()))