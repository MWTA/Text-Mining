# -*- coding: utf-8 -*-

"""
    Software: Example 01
    Description:
    Author: RodriguesFAS
    Date: 23/06/2018
    Contact: <franciscosouaacer@gmail.com>
"""
# 1º Step
from pywsd.lesk import simple_lesk

sent = 'I went to the bank to deposit my money'
ambiguous = 'bank'
answer = simple_lesk(sent, ambiguous, pos='n')

print answer
print answer.definition()

# 2º Step
from pywsd import disambiguate
from pywsd.similarity import max_similarity as maxsim

text = 'I went to the bank to deposit my money'
disambiguate(text)
disambiguate(text, algorithm=maxsim, similarity_option='wup', keepLemmas=True)

# 3º Step
from pywsd.lesk import cached_signatures
cached_signatures['dog.n.01']['simple']
cached_signatures['dog.n.01']['adapted']

# 4º Step
from nltk.corpus import wordnet as wn
wn.synsets('dog')[0]
dog = wn.synsets('dog')[0]
dog.name()
cached_signatures[dog.name()]['simple']
