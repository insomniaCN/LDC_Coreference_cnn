#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import re
import os
import numpy as np
from collections import defaultdict
from util import read_all_lines
from embedding import word2vec


def init_position_voc(position_voc, word_count, index):
    """
    初始化位置字典

    Args:
        position_voc: set, 相对距离集合
        word_count:　int, 句子中的词语数
        index:　int, 事件１的下标
    """
    for i in range(word_count):
        position_voc.add(i-index)
    return 1


def positive_or_negative(path_train, path_test, rel_type=1):
    """
    分别对语料不同类别进行统计

    Args:
        path_train: str, 测试样本路径
        path_test: str, 测试样本路径
        rel_type: int, [0,1]
    Returns:
        path_train: str, 训练样本文件路径
        path_test: str, 测试样本文件路径
        rel_type: int, [0,1]
    """
    if rel_type:
        path_train = os.path.join(path_train + 'train_pos.txt')
        path_test = os.path.join(path_test + 'test_pos.txt')
    else:
        path_train = os.path.join(path_train + 'train_neg.txt')
        path_test = os.path.join(path_test + 'test_neg.txt')
        rel_type = 0

    return path_train, path_test, rel_type


def data_count(path_train, path_test):
    """
    统计语料中所有的词，位置，类别，以及单句最大长

    Args:
        path_train: str, 训练语料的位置
        path_test: str, 测试语料的位置
    Returns:
        word_dict: defaultdict, 词
        position_dict: defaultdict, 位置
        type_dict: defaultdict, 类型
        max(max_len): int, 单句最大长
    """
    max_len = set()
    word_voc, position_voc, type_voc = defaultdict(int), set(), set([0, 1])

    positive_train, positive_test, rel_type = positive_or_negative(path_train, path_test, 1)
    negative_train, negative_test, rel_type = positive_or_negative(path_train, path_test, 0)  # 负例
    positive_lines = read_all_lines(positive_train) + read_all_lines(positive_test)
    negative_lines = read_all_lines(negative_train) + read_all_lines(negative_test)
    all_lines = positive_lines + negative_lines
    del positive_lines, negative_lines

    for i in range(len(all_lines)):
        init_voc(all_lines[i], word_voc, position_voc, max_len)
    word_dict = defaultdict()
    word_voc = sorted(word_voc.items(), key=lambda d: d[1], reverse=True)
    for item in enumerate(word_voc):  # 从１开始编号 
        word_dict[item[1][0]] = item[0] + 1
    position_dict = defaultdict()
    for item in enumerate(sorted(position_voc)):  # 从1开始编号
        position_dict[item[1]] = item[0] + 1
    type_dict = defaultdict()
    for item in enumerate(sorted(type_voc)):
        type_dict[item[1]] = item[0] + 1

    return word_dict, position_dict, type_dict, max(max_len)


def init_voc(corpus_line, word_voc, position_voc, max_len):
    """
    对单行语料进行统计

    Args:
        corpus_line: str, 一行语料
        word_voc: defaultdict, 词及词频
        position_voc: set, 位置词典
        max_len: set, 句长集合
    """
    corpus_line = re.sub('\s+', ' ', corpus_line)
    item = corpus_line.split('|')
    pos = int(item[4])
    words = item[5].split(' ')
    for word in words:
        word_voc[word] += 1
    init_position_voc(position_voc, len(words), pos)
    max_len.add(len(words))


def format_sentence(words, word_dict, max_len):
    """
    对每句话进行填充，不足补0

    Args:
        words: list, 句子切分后的词列表
        word_dict: defaultdict, 词字典
        max_len: int, 最大句长
    Return:
        np.array(result): array, 统计每句话中所包含的字
    """
    result = [0] * max_len  # 用0填充不包含的
    for i in range(len(words)):
        result[i] = word_dict[words[i]]
    return np.array(result)


def get_positions(words_len, max_len, index, position_dict, pos_dict_len):
    """
    对位置信息进行填充，不足的补0

    Args:
        words_len: int, 句子长度
        max_len: int, 句子最大长
        index: int, 事件词下标
        position_dict: defaultdict, 位置字典
        pos_dict_len: int, 位置词典最大长度
    Returns:
        positions: list,
    """
    positions = np.zeros((max_len), dtype='int32')
    for i in range(words_len):
        positions[i] = position_dict[i-index]
    return positions


def get_pos_embed_weights(position_voc, pos_embed_dim=50):
    """
    初始化位置向量

    Args:
        position_voc: dict, 位置字典
        pos_embed_dim: int, 位置特征的维度
    Returns:
        pos_embed_weights: shape(n_symbols,pos_embed_dim)
    """
    n_symbols = len(position_voc.items()) + 1  # 留一个位置给padding的值，即0
    pos_embed_weights = np.zeros((n_symbols, pos_embed_dim), dtype="float32")
    for item in position_voc.values():
        pos_embed_weights[item] = np.random.uniform(-1, 1, (pos_embed_dim,))
    return pos_embed_weights


def get_word_embed_weights(word2vec_model, word_voc):
    """
    初始化word embedding weights

    Args:
        word2vec_model: 词向量模型
        word_voc: 单词字典
    Returns:
        word_embed_weights: shape(n_symbols, word_dim)
    """
    # count = 0
    # for k,v in word_voc.items():
        # if count < 20:
        # print(k, v)
            # count += 1
    # print(count)
 
    word_dim = 300
    n_symbols = len(word_voc.items()) + 1  # 留一个位置给padding的值，即0
    word_embed_weights = np.zeros((n_symbols, word_dim), dtype='float32')
    for word, index in word_voc.items():
        if word in word2vec_model:
            word_embed_weights[index] = word2vec_model[word]
        else:  # 随机初始化词向量
            word_embed_weights[index] = np.random.uniform(-1, 1, (word_dim,))

    return word_embed_weights


def return_voc(corpus_line):
    """
    处理单行语料

    Args:
        corpus_line: str, 单行语料
    Returns:
        pos: int, 位置
        words: list, 词
    """
    corpus_line = re.sub('\s+', ' ', corpus_line)
    item = corpus_line.split('|')
    pos = int(item[4])
    words = item[5].split(' ')
    return pos, words


def load_data_nostatic(pos_embed_dim=50):
    """
    加载数据，动态词向量

    Args:
        pos_embed_dim: int, 位置维度
    Returns:
        data_sentences: np.array(example_pairs,max_len*2)
        data_positions: np.array(example_pairs,max_len*2)
        labels: 事件对标签
        word_embed_weights: 初始化词向量
        pos_embed_weights: 初始化位置向量
        type_voc: 标签类型
        max_len: 最长句
    """
    path_train = './LDC_corpus/corpus_handle/'
    path_test = './LDC_corpus/corpus_handle/'
    word_voc, position_voc, type_voc, max_len = data_count(path_train, path_test)
    positive_train, positive_test, rel_type = positive_or_negative(path_train, path_test, 1)
    negative_train, negative_test, rel_type = positive_or_negative(path_train, path_test, 0)  # 负例
    positive_lines = read_all_lines(positive_train) + read_all_lines(positive_test)
    negative_lines = read_all_lines(negative_train) + read_all_lines(negative_test)
    negative_lines = negative_lines[:25000]

    sentence_count = len(positive_lines + negative_lines)  # 语料总行数
    fst_data_sentences = np.zeros((int(sentence_count/2), max_len), dtype='int32')  # 事件1所在句子
    fst_data_positions = np.zeros((int(sentence_count/2), max_len), dtype='int32')  # 位置1
    sec_data_sentences = np.zeros((int(sentence_count/2), max_len), dtype='int32')  # 事件2所在句子
    sec_data_positions = np.zeros((int(sentence_count/2), max_len), dtype='int32')  # 位置2
    labels = []  # 标签
    # 初始化word look-up table, 利用pre_train的词向量
    word2vec_model = word2vec(binary=True)
    word_embed_weights = get_word_embed_weights(word2vec_model, word_voc)
    # 初始化posiiton look-up table,　随机初始化
    pos_embed_weights = get_pos_embed_weights(position_voc, pos_embed_dim=pos_embed_dim)
    # 处理positive语料
    for i in range(0, len(positive_lines), 2):
        fst_pos, fst_words = return_voc(positive_lines[i])
        sec_pos, sec_words = return_voc(positive_lines[i+1])
        rel_type = 1
        fst_sentence = format_sentence(fst_words, word_voc, max_len)  # 词转换成id
        sec_sentence = format_sentence(sec_words, word_voc, max_len)
        fst_data_sentences[int(i/2), :] = fst_sentence
        sec_data_sentences[int(i/2), :] = sec_sentence
        fst_data_positions[int(i/2), :] = get_positions(len(fst_words), max_len, fst_pos, 
                                                      position_voc, len(position_voc.items()))
        sec_data_positions[int(i/2), :] = get_positions(len(sec_words), max_len, sec_pos, 
                                                          position_voc, len(position_voc.items()))
        labels.append(type_voc[rel_type])  # 类别id
    positive_pairs = int(len(positive_lines) / 2)
    # 处理negative语料
    for i in range(0, len(negative_lines), 2):
        fst_pos, fst_words = return_voc(negative_lines[i])
        sec_pos, sec_words = return_voc(negative_lines[i+1])
        rel_type = 0
        fst_sentence = format_sentence(fst_words, word_voc, max_len)  # 词转换成id
        sec_sentence = format_sentence(sec_words, word_voc, max_len)
        fst_data_sentences[int(i/2)+positive_pairs, :] = fst_sentence
        sec_data_sentences[int(i/2)+positive_pairs, :] = sec_sentence
        fst_data_positions[int(i/2)+positive_pairs, :] = get_positions(len(fst_words), max_len, 
                                                                     fst_pos, position_voc, 
                                                                     len(position_voc.items()))
        sec_data_positions[int(i/2)+positive_pairs, :] = get_positions(len(sec_words), max_len, 
                                                                         sec_pos, position_voc, 
                                                                         len(position_voc.items()))
        labels.append(type_voc[rel_type])  # 类别id

    return fst_data_sentences, sec_data_sentences, fst_data_positions, sec_data_positions, labels, word_embed_weights, pos_embed_weights, \
        type_voc, max_len


if __name__ == '__main__':
    data_sentences, data_positions, labels, word_embed_weights, pos_embed_weights, \
        type_voc, max_len = load_data_nostatic(50)
