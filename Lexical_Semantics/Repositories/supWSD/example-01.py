"""
    Software: Example 01
    Description: Recupere o SupSense para todas as palavras.
    Date: 23/06/2018
    Author: RodriguesFAS
    Contact: <franciscosouzaacer@gmail.com>
"""

from supwsd.wsd import SupWSD
		
text = 'The human brain is quite proficient at word-sense disambiguation.'

for sense in SupWSD().senses(text):
	print('Word: {}\tLemma: {}\tPOS: {}\tSense: {}'.format(sense.word, sense.lemma, sense.pos, sense.key()))