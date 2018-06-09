#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Description: Problema 02.
    Date: 17/05/2018
    Author: RodriguesFAS
    Email: franciscosouzaacer@gmail.com
    License: MIT

    Dependency:
        <https://github.com/amueller/word_cloud>

    Tutorial:
        <http://minerandodados.com.br/index.php/2017/09/04/cafe-com-codigo-nuvem-de-tags/>
        <http://neylsoncrepalde.github.io/2017-03-20-nuvens-de-palavras-dinamicas/s>

    Tags Words:
        <https://cs.nyu.edu/grishman/jet/guide/PennPOS.html>

    Stopwords:
        <http://xpo6.com/list-of-english-stop-words/>
        <http://www.lextek.com/manuals/onix/stopwords1.html>

    Stemming and Lemmatization
        <https://blog.bitext.com/what-is-the-difference-between-stemming-and-lemmatization/>

    Tree Constituency Parsing
        <https://www.commonlounge.com/discussion/1ef22e14f82444b090c8368f9444ba78>

    
"""


import nltk
import matplotlib.pyplot as plt

from nltk import FreqDist
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer


path_corpus_1 = '/home/rodriguesfas/Workspace/Mining-Text/MT_2018_PPGIA/AULA_07/LE_05/NLP.txt'
path_corpus_2 = '/home/rodriguesfas/Workspace/Mining-Text/MT_2018_PPGIA/AULA_07/LE_05/Corpus_en_NER.txt'
path_stopwords = '/home/rodriguesfas/Workspace/Mining-Text/MT_2018_PPGIA/AULA_07/LE_05/stopwords.txt'
path_sent = '/home/rodriguesfas/Workspace/Mining-Text/MT_2018_PPGIA/AULA_07/LE_05/sent.txt'

TEST = True


def loadCorpus(path):
    LOG('Carregando corpus...')
    return open(path).read()


def tokenizer(corpus):
    LOG('Tokenizer corpus...')
    tokens = nltk.word_tokenize(corpus.lower())
    return [word for word in tokens if word.isalpha()]


def words_frequency(corpus, frequency):
    LOG('Calculando frequência das palavras...')
    freq_dist = nltk.FreqDist(tokenizer(corpus))
    return freq_dist.most_common(frequency)


def classify_words(corpus):
    LOG('Tags corpus..')
    list_token = tokenizer(corpus)
    words_list = nltk.pos_tag(list_token)
    return words_list


def generates_list_of_words_class(words_list_all_tags, tag):
    LOG('Generate list of words tags...')
    words_list_tag_selected = []

    # Monta uma lista de palavras conforme a classe selecionada.
    for y in xrange(len(words_list_all_tags)):
        if(words_list_all_tags[y][1] == tag):
            word_tag_selected = words_list_all_tags[y]
            words_list_tag_selected.append(word_tag_selected)

    return words_list_tag_selected


def remove_stopwords(corpus):
    LOG('Remove stopwords..')

    list_stopwords = loadCorpus(path_stopwords)
    words_filtered = corpus[:]
    words_filtered = [i for i in corpus.split() if not i in list_stopwords]

    return (" ".join(words_filtered))


def words_lemmatizing(corpus):
    LOG('Lemmatizing..')
    lemmatizer = WordNetLemmatizer()

    return lemmatizer.lemmatize(corpus)


def words_substantive(corpus, stopwords, lemmatizing):
    LOG('Classification substantive..')

    words_list_tag_selected = generates_list_of_words_class(
        classify_words(corpus), 'NN')

    # remove caracteres especiais.
    words_list_substantive = str(words_list_tag_selected).replace('[', '').replace(']', '').replace(
        '(', '').replace(')', '').replace(',', '').replace('NN', '').replace("'", '')

    if stopwords is True:
        words_list_substantive = remove_stopwords(words_list_substantive)

    if lemmatizing is True:
        words_list_substantive = words_lemmatizing(words_list_substantive)

    return words_list_substantive


def words_verbs(corpus, stopwords, lemmatizing):
    LOG('Classification verbs...')

    words_list_tag_selected = generates_list_of_words_class(
        classify_words(corpus), 'VB')

    # remove caracteres especiais.
    words_list_verbs = str(words_list_tag_selected).replace('[', '').replace(']', '').replace(
        '(', '').replace(')', '').replace(',', '').replace('VB', '').replace("'", '')

    if stopwords is True:
        words_list_verbs = remove_stopwords(words_list_verbs)

    if lemmatizing is True:
        words_list_verbs = words_lemmatizing(words_list_verbs)

    return words_list_verbs


def cloud_of_words(corpus, value):
    LOG('Config cloudwords...')
    wordcloud = WordCloud(background_color='white', max_font_size=200,
                          width=1520, height=535, scale=3, max_words=value).generate(corpus)
    return generate(wordcloud)


def generate(wordcloud):
    LOG('Generate wordcloud...')

    plt.figure(figsize=(16, 9))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


def tree_parsing_constituency(sent):
    LOG('Tree Parsing Constituency...')
    LOG(str(sent))

    groucho_grammar = nltk.CFG.fromstring("""
        S -> NP VP
        PP -> P NP
        NP -> Det N | Det N PP | 'I'
        VP -> V NP | VP PP
        Det -> 'to' | 'my'
        N -> 'market' | 'shorts'
        V -> 'went'
        P -> 'in'
    """)

    parser = nltk.ChartParser(groucho_grammar)

    for tree in parser.parse(sent):
        print(tree)


def LOG(text):
    if TEST is True:
        print(">> " + text)



def main():
    LOG('Stating...')

    # 2.1
    # Gera nuvem de palavras contendo apenas os lemas dos 20 substântivos mais frequentes.
    # print(words_substantive(loadCorpus(path_corpus_1), False))
    # cloud_of_words(words_substantive(loadCorpus(path_corpus_1), False, True), 20)

    # 2.2
    # Gera nuvem de palavras contendo apenas os lemas dos 20 verbos mais frequantes.
    # print(words_verbs(loadCorpus(path_corpus_1)))
    # cloud_of_words(words_verbs(loadCorpus(path_corpus_1), False, True), 20)

    # 2.3
    # Gera nuvem de palavras removendo stopwords contendo apenas os lemas das 20 palavras
    #  mais frequentes, substantivos ou verbos.
    # 2.3.1
    #cloud_of_words(words_substantive(loadCorpus(path_corpus_2), True, True), 20)
    # 2.3.2
    # cloud_of_words(words_verbs(loadCorpus(path_corpus_2), True, True), 20)

    # 2.4
    tree_parsing_constituency(loadCorpus(path_sent).split())


    # Run
main()
LOG('Finished!')
