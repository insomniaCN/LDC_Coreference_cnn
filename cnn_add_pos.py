#!/usr/bin/env python
# -*- encoding: utf-8 -*-


from time import time
import numpy as np
from dataset import load_data_nostatic
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.layers import Input, merge
from keras.layers import Convolution1D, GlobalMaxPooling1D, Embedding
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils.visualize_util import plot

# 参数设置
BATCH_SIZE = 50
NB_EPOCH = 20
WINDOW_SIZE = 3
NB_FILTERS = 150
WORD_EMBED_DIM = 300
POS_EMBED_DIM = 50
NB_CLASSES = 2
hidden_dims = 32

t0 = time()
# 加载数据，训练：测试 = 97876:58832
print("loading data...")
bound = 120000   # 训练数据数量
fst_data_sentences, sec_data_sentences, fst_data_positions, sec_data_positions, \
    labels, word_embed_weights, pos_embed_weights, type_voc, max_len = \
    load_data_nostatic(pos_embed_dim=POS_EMBED_DIM)
labels = np.array(labels, dtype='int32')
print("label shape: ", labels.shape)
labels = np_utils.to_categorical(labels, nb_classes=NB_CLASSES)
np.random.seed(12345)
np.random.shuffle(fst_data_sentences)
np.random.seed(12345)
np.random.shuffle(sec_data_sentences)
np.random.seed(12345)
np.random.shuffle(fst_data_positions)
np.random.seed(12345)
np.random.shuffle(sec_data_positions)
np.random.seed(12345)
np.random.shuffle(labels)
fst_train_data_sentences, sec_train_data_sentences, fst_train_data_positions, \
    sec_train_data_positions, train_labels = \
    fst_data_sentences[:bound], sec_data_sentences[:bound], fst_data_positions[:bound], \
    sec_data_positions[:bound], labels[:bound]
fst_test_data_sentences, sec_test_data_sentences, fst_test_data_positions, \
    sec_test_data_positions, test_labels = \
    fst_data_sentences[bound:], sec_data_sentences[bound:], fst_data_positions[bound:], \
    sec_data_positions[bound:], labels[bound:]

print("word embedd dim shape: ", word_embed_weights.shape)
print("X sentences train shape: ", fst_train_data_sentences.shape)
print("X positions train shape: ", fst_train_data_positions.shape)
print("Y train shape:", len(train_labels))
print("X senteces test shape:", fst_test_data_sentences.shape)
print("X positions test shape:", fst_test_data_positions.shape)

# 构建模型
print("build model...")
# embedding individually for event1, event2
input_fst_sentence = Input(
    shape=(
        max_len,
    ),
    dtype='int32',
    name='input_fst_sentence')
embedding_fst_sentence = Embedding(input_dim=word_embed_weights.shape[0],
                                   output_dim=WORD_EMBED_DIM,
                                   weights=[word_embed_weights],
                                   input_length=max_len,
                                   trainable=False,
                                   # dropout=0.5,
                                   name="embedding_fst_sentence")(input_fst_sentence)
input_fst_position = Input(
    shape=(
        max_len,
    ),
    dtype='int32',
    name='input_fst_position')
embedding_fst_position = Embedding(input_dim=pos_embed_weights.shape[0],
                                   output_dim=POS_EMBED_DIM,
                                   weights=[pos_embed_weights],
                                   input_length=max_len,
                                   trainable=True,
                                   name='embedding_fst_position')(input_fst_position)
# input_fst = tf.concat(values=[embedding_fst_sentence, embedding_fst_position], axis=2)
input_fst = merge([embedding_fst_sentence, embedding_fst_position],
                  mode='concat',
                  concat_axis=2)  # event 1
input_sec_sentence = Input(
    shape=(
        max_len,
    ),
    dtype='int32',
    name='input_sec_sentence')
embedding_sec_sentence = Embedding(input_dim=word_embed_weights.shape[0],
                                   output_dim=WORD_EMBED_DIM,
                                   weights=[word_embed_weights],
                                   input_length=max_len,
                                   trainable=False,
                                   # dropout=0.5,
                                   name="embedding_sec_sentence")(input_sec_sentence)
input_sec_position = Input(
    shape=(
        max_len,
    ),
    dtype='int32',
    name='input_sec_position')
embedding_sec_position = Embedding(input_dim=pos_embed_weights.shape[0],
                                   output_dim=POS_EMBED_DIM,
                                   weights=[pos_embed_weights],
                                   input_length=max_len,
                                   trainable=True,
                                   name='embedding_sec_position')(input_sec_position)
# input_sec = tf.concat(values=[embedding_sec_sentence, embedding_sec_position], axis=2)
input_sec = merge([embedding_sec_sentence, embedding_sec_position],
                  mode='concat',
                  concat_axis=2)  # event 2

# convolution/maxpooling layer for e1, e2 individually
cnn_layer_fst = Convolution1D(nb_filter=NB_FILTERS,
                              filter_length=WINDOW_SIZE,
                              activation='relu',
                              border_mode='valid',
                              subsample_length=1
                              )(input_fst)
cnn_layer_sec = Convolution1D(nb_filter=NB_FILTERS,
                              filter_length=WINDOW_SIZE,
                              activation='relu',
                              border_mode='valid',
                              subsample_length=1
                              )(input_sec)
maxpooling_layer_fst = GlobalMaxPooling1D()(cnn_layer_fst)
maxpooling_layer_sec = GlobalMaxPooling1D()(cnn_layer_sec)

# add hidden layer for result input individually
hidden_output_fst = Dense(output_dim=hidden_dims,
                          activation='tanh')(maxpooling_layer_fst)
hidden_output_sec = Dense(output_dim=hidden_dims,
                          activation='tanh')(maxpooling_layer_sec)

# event-pairs concat
#pair_input = tf.concat(values=[hidden_output_fst, hidden_output_sec], axis=0)
pair_input = merge([hidden_output_fst, hidden_output_sec],
                   mode='concat',
                   concat_axis=-1)
X_dropout = Dropout(0.5)(pair_input)
X_output = Dense(NB_CLASSES,
                 W_regularizer=l2(0.1),
                 activation='sigmoid')(X_dropout)
model = Model(input=[input_fst_sentence, input_fst_position,
                     input_sec_sentence, input_sec_position],
              output=[X_output])
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("train...")
model_path = "./model/18w_40_model.hdf5"
#model_path = './model/shuffle_40_model.hdf5'  # 句长40, 负例随机选取1w条作为训练集
modelcheckpoint = ModelCheckpoint(model_path,
                                  verbose=1,
                                  save_best_only=True)
model.fit([fst_train_data_sentences, fst_train_data_positions,
           sec_train_data_sentences, sec_train_data_positions],
          [train_labels],
          # 梯度下降时每一个batch所包含的样本数
          batch_size=BATCH_SIZE,
          nb_epoch=NB_EPOCH,
          callbacks=[modelcheckpoint],
          validation_data=([fst_test_data_sentences, fst_test_data_positions,
                            sec_test_data_sentences, sec_test_data_positions],
                           [test_labels]))
plot(model, to_file="./model/model.png")
print("Done! time costs: ", time() - t0)
