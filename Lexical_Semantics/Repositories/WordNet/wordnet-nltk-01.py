#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import nltk
# nltk.download('wordnet')

# primeiro você importa o wordnet.
from nltk.corpus import wordnet

# usamos um termo "program" para encontrar synsets.
syns = wordnet.synsets("program")

# exemplos de saídas.
print(syns[0].name())
print(syns[0].lemmas()[0].name())
print(syns[0].definition())
print(syns[0].examples())

# aqui pode-se discernir sinônimo e antônimo.. os lemas serão sinônimos, e então você pode
# usa antônimo para encontrar s antônimos para os lemas. Como tal, pode-se preencher algumas
# listas como:
synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

# Os resultados mostrão que 
print(set(synonyms))
print(set(antonyms))

# comparando o susbstantivo "ship" e "boat".
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('cat.n.01')
print(w1.wup_similarity(w2))
