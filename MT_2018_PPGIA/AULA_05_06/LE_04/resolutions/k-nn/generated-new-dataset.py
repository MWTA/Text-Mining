#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Software: Generated New Dataset
    
    Description: Gera um novo dataset organizado em um único arquivo.
        
        Passos:

        - 1º Ler todos os arquivos .TXT dos diretórios neg_ e pos_ do dataset em uso.
        - 2º Converte dodas as letras para minúsculo.
        - 3º Remove caracteres numéricos.
        - 4º Remove caracteres especiais.
        - 5º Adiciona Rótulos as Instâncias.
        - 6º Limpa sujeira criada durante rotulagem.
        - 7º Adicionar documento ao novo arquivo.
        - 8º Salva o arquivo.
        - 9º Ler o arquivo salvo e raliza um split de acordo com as frações definidas.
        - 10º Salva os arquivos.

    Date: 31/05/2018
"""

import re
import os
import sys


reload(sys)
sys.setdefaultencoding('utf-8')

path_root = '/home/rodriguesfas/Workspace/k-nn/data/'

dir_dataset = 'polarity-detection-200-reviews/'
dir_neg = 'neg/'
dir_pos = 'pos/'
dir_out = 'out/'

path_files_neg = path_root + dir_dataset + dir_neg
path_files_pos = path_root + dir_dataset + dir_pos
path_dataset_output = path_root + dir_out + 'generated-polarity-data.csv'

"""
    Rótulos usados para classificar as instâncias.
    0: instâncias negativas
    1: instâncias positivas
"""
label_neg = [0]
label_pos = [1]

"""
    Debug Console
    Flag
        True 
        False
"""
TEST = True


def LOG(text):
    if TEST is True:
        print(">> " + text)


def remove_numbers(document):
    LOG('Remove numbers..')
    return re.sub('[-|0-9]', ' ', document).strip()


def remove_special_characters(document):
    LOG('Remove characters special..')
    document = re.sub(r'[-_./?!,`":;=+()<>|@#$%&*^~\']', '', document)
    document = "".join(document.splitlines())
    document = ' '.join(document.split())
    return document.strip()


def clean(document):
    document = document.replace('(', '').replace(')', '').replace('[', '').replace(
        ']', '').replace("'", "").replace("[0]", "0").replace("[1]", "1").replace("\\", "")
    return document.strip()


def load_files(path_files, label):
    dataset_output = open(path_dataset_output, 'a+')
    for file in os.listdir(path_files):
        LOG('Processing data file: ' + file)
        if file.endswith('.txt'):
            document = open(path_files + file).read().lower()
            document = remove_numbers(document)
            document = remove_special_characters(str(document))
            LOG('Labeling content..')
            document = document, label
            document = clean(str(document))
            dataset_output.write(document + '\n')
    dataset_output.close()
    LOG('Dataset generated-polarity-data.csv generated!')


def split_dateset(path_dataset, split):
    LOG('Split dataser ...')
    nfile = 1
    cont = 0
    split = split
    with open(path_dataset) as documents:
        for doc in documents:
            arq = open(path_root + dir_out + "split/" +
                       "split" + str(nfile) + ".csv", 'a+')
            doc_temp = clean(
                str([doc.split(",")[0], doc.split(",")[1].strip()]))
            arq.write(str(doc_temp) + '\n')
            cont += 1
            if cont == 40:
                cont = 0
                nfile += 1


def main():
    LOG("Start!")

    LOG('Processing directory neg_ ...')
    load_files(path_files_neg, label_neg)

    LOG('Processing directory pos_ ...')
    load_files(path_files_pos, label_pos)

    split_dateset(path_dataset_output, 5)

    LOG("Finalized!")

    # Run
if __name__ == '__main__':
    main()
