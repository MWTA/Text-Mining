"""
    Software: Example 03
    Description: Recuperar todos os objetos SupResult (com chaves de sentido e probabilidades)
    Date: 23/06/2018
    Author: RodriguesFAS
    Contact: <franciscosouzaacer@gmail.com>
"""

from supwsd.wsd import SupWSD

text = 'The human brain is quite proficient at word-sense disambiguation.'

for sense in SupWSD().senses(text,True):
	print('Word: {}\tLemma: {}\tPOS: {}\tSense: {}'.format(sense.word, sense.lemma, sense.pos, sense.key()))
	for result in sense.results:
		print('Sense {}\tProbability: {}'.format(result.key, result.prob))