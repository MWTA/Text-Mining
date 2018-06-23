"""
    Software: Example 02
    Description: Recupere o SupSense para palavras específicas (use SupWSD.SENSE_TAG para marcar uma palavra)
    Date: 23/06/2018
    Author: RodriguesFAS
    Contact: <franciscosouzaacer@gmail.com>
"""

from supwsd.wsd import SupWSD
		
text = 'The human ' + SupWSD.SENSE_TAG + 'brain' + SupWSD.SENSE_TAG + ' is quite proficient at word-sense disambiguation. The fact that natural language is formed '+ SupWSD.SENSE_TAG+'in a way'+ SupWSD.SENSE_TAG+' that requires so much of it is a ' + SupWSD.SENSE_TAG + 'reflection' + SupWSD.SENSE_TAG + ' of that neurologic reality.'

for sense in SupWSD().senses(text):
	print('Word: {}\tLemma: {}\tPOS: {}\tSense: {}'.format(sense.word, sense.lemma, sense.pos, sense.key()))