"""
    Software: Dependencies
    Description: Visualize dependencies and entities in your browser and notebook, or export HTML.
    Author: RodriguesFAS
    Date: 20/06/2018
    Contact: <franciscosouzaacer@gmail.com>
"""

import spacy
from spacy import displacy

nlp = spacy.load('en')
doc = nlp(u'This is a sentence.')
displacy.serve(doc, style='dep')