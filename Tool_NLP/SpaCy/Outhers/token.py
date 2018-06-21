#!/usr/bin/env python
# -*- coding: utf-8 -*-

import spacy

nlp = spacy.load('en')

doc = nlp(u"Next week I'll   be in Madrid.")

for token in doc:
    print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}".format(
        token.text,
        token.idx,
        token.lemma_,
        token.is_punct,
        token.is_space,
        token.shape_,
        token.pos_,
        token.tag_
    ))