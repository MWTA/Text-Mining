from pybabelfy.babelfy import *

text = "BabelNet is both a multilingual encyclopedic dictionary and a semantic network"
lang = "EN"
# This only works for the demo example. Change it for your RESTful key (you must register at babelfy.org for it)
key = "5e962130-b37f-4105-8512-4c97b4f3cb30"

babelapi = Babelfy()

semantic_annotations = babelapi.disambiguate(text, lang, key)
