#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import re
import pickle
from time import time
import numpy as np
from collections import defaultdict
from util import read_all_lines
from gensim.models import KeyedVectors

def word2vec(binary=False):
    """
    选取加载bin文件或者pk文件

    Args:
        binary: Boolean
    Returns:
        None
    """
    if binary:
        return word2vec_bin()
    else:
        return word2vec_pk()


def word2vec_bin():
    """
    从二进制中加载词向量

    Returns:
        model: 加载好的模型
    """
    t0 = time()
    print("load wordvecs...")
    path = '~/wordvectors/EN.GoogleNews.100B.300d.bin'
    model = KeyedVectors.load_word2vec_format(path, binary=True)
    print("done in %d s." % (time()-t0))
    return model


def word2vec_pk():
    """
    用pickle加载模型

    Returns:
        word2vec_model: 加载好的模型
    """
    t0 = time()
    print("load wordvecs...")
    path = '../../wordvectors/EN.GoogleNews.100B.300d.pk'

    file_pk = open(path, 'rb')
    word2vec_model = pickle.load(file_pk)
    file_pk.close()
    print("done in %d s." % (time()-t0))
    return word2vec_model


def word2vec_txt():
    """
    从txt文件中加载词向量

    Returns:
        word2vec_model: 加载好的模型
    """
    t0 = time()
    print("load wordvecs...")
    path = '~/Documents/wordvectors/EN.GoogleNews.100B.300d.txt'
    lines = read_all_lines(path)
    word2vec_model = defaultdict()
    for line in lines[1:]:
        items = line.split(' ')
        word = item[0]
        vec = np.array(item[1:], dtype='float32')
        word2vec_model[word] = vec
    print("done in %d s." % (time()-t0))
    return word2vec_model


def write2pk():
    """
    将txt模型转变成pk模型
    """
    t0 = time()
    print("load wordvecs...")
    path = '../../wordvectors/EN.GoogleNews.100B.300d.txt'
    lines = read_all_lines(path)
    word2vec_model = defaultdict()
    for line in lines[1:]:
        items = line.split(' ')
        word = item[0]
        vec = np.array(item[1:], dtype='float32')
        word2vec_model[word] = vec
    pk_file = open('~/Documents/wordvectors/EN.GoogleNews.100B.300d.pk', 'wb')
    pickle.dump(word2vec_model, pk_file)
    pk_file.close()
    print("done in %d s." % (time()-t0))
