# -*- coding: utf-8 -*-

"""
    Software: POS Tagging
    Tutorial: <https://spacy.io/usage/linguistic-features#section-pos-tagging>
    Description: Marcação de parte da fala
        Após a tokenização, o spaCy pode analisar e marcar um dado Doc. É aí que entra o 
        modelo estatístico, que permite ao spa fazer uma previsão de qual tag ou rótulo 
        provavelmente se aplica neste contexto. Um modelo consiste em dados binários e é 
        produzido mostrando a um sistema exemplos suficientes para fazer previsões que 
        generalizam a linguagem - por exemplo, uma palavra que segue "o" em inglês é 
        provavelmente um substantivo.

        Anotações linguísticas estão disponíveis como Tokenatributos. Como muitas bibliotecas
        NLP, o spaCy codifica todas as strings para valores de hash para reduzir o uso de 
        memória e melhorar a eficiência. Então, para obter a representação de string legível 
        de um atributo, precisamos adicionar um sublinhado _ao seu nome:
    Date: 20/06/2018
    Author: RodriguesFAS
    Contact: <franciscosouzaacer@gmail.com>
"""

import spacy

nlp = spacy.load('en_core_web_sm')

doc = nlp(unicode('Apple is looking at buying U.K. startup for $1 billion').decode('utf8'))

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)