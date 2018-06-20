#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pip install -U spaCy
# python -m spacy download en

import spacy

nlp = spacy.load('en')
doc = nlp(u"Hello     World!")

for token in doc:
    print('"' + token.text + '"')
    print('"' + token.text + '"', token.idx)