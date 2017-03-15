#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import time
import numpy as np
from dataset import load_data_nostatic
from keras.models import Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution1D, GlobalMaxPooling1D, Embedding

# 参数设置
BATCH_SIZE = 50
NB_EPOCH = 50
WINDOW_SIZE = 3
NB_FILTERS = 150
WORD_EMBED_DIM = 300
POS_EMBED_DIM = 50
NB_CLASSES = 2
max_features = 17038
hidden_dims = 250

t0 = time()
# 加载数据，训练：测试 = 97876:58832
print("loading data...")
bound = 18000   # 训练数据数量
fst_data_sentences, sec_data_sentences, fst_data_positions, sec_data_positions, \
    labels, word_embed_weights, pos_embed_weights, type_voc, max_len = \
    load_data_nostatic(pos_embed_dim=POS_EMBED_DIM)

fst_train_data_sentences, sec_train_data_sentences, fst_train_data_positions, \
    sec_train_data_sentences, train_labels = \
    fst_data_sentences[:bound], sec_data_sentences[:bound], fst_data_positions[:bound], \
    sec_data_positions[:bound], labels[:bound]
fst_test_data_sentences, sec_test_data_sentences, fst_test_data_positions, \
    sec_test_data_positions, test_labels = \
    fst_data_sentences[bound:], sec_data_sentences[bound:], fst_data_positions[bound:], \
    sec_data_positions[bound:], labels[bound:]
print("word embedd dim shape: ", word_embed_weights.shape)
print("X train shape:", fst_train_data_sentences.shape)
print("Y train shape:", len(train_labels))
print("X train shape:", fst_test_data_sentences.shape)
# 构建模型
print("build model...")
# embedding_layer = Embedding(max_features,
#                             WORD_EMBED_DIM,
#                             weights=[word_embed_weights],
#                             input_length=max_len * 2,
#                             trainable=False,
#                             dropout=0.5)
# sequence_input = Input(shape=(max_len * 2, ), dtype='int32')
# embedded_sequence = embedding_layer(sequence_input)
# x = Conv1D(nb_filter=NB_FILTERS,
#            filter_length=WINDOW_SIZE,
#            activation='relu')(embedded_sequence)
# x = GlobalMaxPooling1D()(x)
# x = Dense(1)(x)
# x = Dropout(0.2)(x)
# x = Activation('relu')(x)
#
# model = Model(sequence_input)
model = Model()
model.add(Embedding(max_features,
                    WORD_EMBED_DIM,
                    weights=[word_embed_weights],
                    input_length=max_len * 2,
                    trainable=False))
model.add(Convolution1D(nb_filter=NB_FILTERS,
                        filter_length=WINDOW_SIZE,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(1))
model.add(Dropout(0.5))
model.add(Activation('relu'))
# model.add(Dense(1))
# model.add(Activation("sigmoid"))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(train_data_sentences, train_labels,
          batch_size=BATCH_SIZE,
          nb_epoch=NB_EPOCH,
          validation_data=(test_data_sentences, test_labels))
