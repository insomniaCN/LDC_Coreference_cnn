#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function, division
from dataset import *
from keras.models import load_model
import numpy as np
from keras.utils import np_utils
import os
from evaluate import compute

def load_data_nostatic_test(pos_embed_dim=50):
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
    positive_lines = read_all_lines(positive_test)
    negative_lines = read_all_lines(negative_test)
    # 随机选取5000个负例样本
    # neg_pair_lines = []
    # for i in range(0, len(negative_lines), 2):
    #     neg_pair_lines.append([negative_lines[i], negative_lines[i+1]])
    #     np.random.shuffle(neg_pair_lines)
    # neg_pair_lines = neg_pair_lines[:10000]  # 对负例样本打乱随机选取5000个样本
    # negative_lines = []
    # for line in neg_pair_lines:
    #     negative_lines.append(line[0])
    #     negative_lines.append(line[1])
    # del neg_pair_lines

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
        rel_type = "coreference"
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
        rel_type = "non-coreference"
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

    return fst_data_sentences, sec_data_sentences, fst_data_positions, sec_data_positions, labels, type_voc


if __name__ == "__main__":
    fst_data_sentences, sec_data_sentences, fst_data_positions, sec_data_positions, labels, type_voc = \
        load_data_nostatic_test(50)
    labels = np.array(labels, dtype='int32')
    labels = np_utils.to_categorical(labels, nb_classes=2)
    print("X sentences shape: ", fst_data_sentences.shape)
    print("X positions shape: ", fst_data_positions.shape)
    print("labels shape: ", labels.shape)
    model = load_model('./model/18w_40_model.hdf5')
    pred = model.predict(
        x=[fst_data_sentences, fst_data_positions, 
           sec_data_sentences, sec_data_positions],
        verbose=0,
        batch_size=50
        )
    pre_labels = []
    for p in pred:
        pre_labels.append(p.argmin())
    print(pre_labels)
    right_labels = []
    for r in labels[:len(pre_labels)]:
        right_labels.append(r.argmin())

    root = './result'
    if not os.path.exists(root):
        os.mkdir(root)
    result_path = root + '/18w_40_result.txt'
    acc, pre, f = compute(pre_labels, right_labels, -1, type_voc, result_path)
    print(acc, pre, f)
