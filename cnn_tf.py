#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import division, print_function
import sys
sys.path.append("./TFNN/layers/")
sys.path.append("./TFNN/")
from tqdm import tqdm
from time import time
import numpy as np
import tensorflow as tf
from dataset import load_data_nostatic
from Embedding import Embedding
from Convolutional1D import Convolutional1D
from Dense import Dense
from Dropout import Dropout
from evaluate import sim_compute


# 参数设置
BATCH_SIZE = 50
NB_EPOCH = 50
WINDOW_SIZE = 3
NB_FILTERS = 150
WORD_EMBED_DIM = 300
POS_EMBED_DIM = 50
NB_CLASSES = 2
hidden_dim = 75


start_time = time()
# 加载数据，训练：测试 = 97876:58832
print("loading data...")
bound = 15000  # 训练数据数量
fst_data_sentences, sec_data_sentences, fst_data_positions, sec_data_positions, \
    labels, word_embed_weights, pos_embed_weights, type_voc, max_len = \
    load_data_nostatic(pos_embed_dim=POS_EMBED_DIM)
fst_train_data_sentences, sec_train_data_sentences, \
    fst_train_data_positions, sec_train_data_positions, \
    train_labels = \
    fst_data_sentences[:bound], sec_data_sentences[:bound], \
    fst_data_positions[:bound], sec_data_positions[:bound], \
    labels[:bound]
fst_valid_data_sentences, sec_valid_data_sentences, \
    fst_valid_data_positions, sec_valid_data_positions, \
    valid_labels = \
    fst_data_sentences[bound:], sec_data_sentences[bound:], \
    fst_data_positions[bound:], sec_data_positions[bound:], \
    labels[bound:]
print("data_senteces shape:", fst_train_data_sentences.shape)
print("data_positions shape:", fst_train_data_positions.shape)
print("max length: ", max_len)

# 构建模型
print("build model...")
input_fst_word_ph = tf.placeholder(
    tf.int32,
    shape=(None, max_len)
)
input_sec_word_ph = tf.placeholder(
    tf.int32,
    shape=(None, max_len)
)
input_fst_pos_ph = tf.placeholder(
    tf.int32,
    shape=(None, max_len)
)
input_sec_pos_ph = tf.placeholder(
    tf.int32,
    shape=(None, max_len)
)
label_ph = tf.placeholder(
    tf.int32,
    shape=(None)
)

# Embedding
fst_word_embed_weights = tf.Variable(
    word_embed_weights,
    tf.float32
)  # word embedding
sec_word_embed_weights = tf.Variable(
    word_embed_weights,
    tf.float32
)
fst_word_embed_layer = Embedding(
    params=fst_word_embed_weights,
    ids=input_fst_word_ph
)
sec_word_embed_layer = Embedding(
    params=sec_word_embed_weights,
    ids=input_sec_word_ph
)

fst_pos_embed_weights = tf.Variable(
    pos_embed_weights,
    tf.float32
)  # pos embedding
sec_pos_embed_weights = tf.Variable(
    pos_embed_weights,
    tf.float32
)
fst_pos_embed_layer = Embedding(
    params=fst_pos_embed_weights,
    ids=input_fst_pos_ph
)
sec_pos_embed_layer = Embedding(
    params=sec_pos_embed_weights,
    ids=input_sec_pos_ph
)

# Merge
fst_word_pos_output = tf.concat(  # 拼接词和位置
    values=[fst_word_embed_layer.output,
            fst_pos_embed_layer.output],
    axis=2  # 拼接维度
)
sec_word_pos_output = tf.concat(
    values=[sec_word_embed_layer.output,
            sec_pos_embed_layer.output],
    axis=2
)

# Convolutional&MaxPooling layer
conv_layer_fst = Convolutional1D(
    input_data=fst_word_pos_output,
    filter_length=WINDOW_SIZE,
    nb_filter=NB_FILTERS,
    activation='relu',
    pooling=True
)
conv_layer_sec = Convolutional1D(
    input_data=sec_word_pos_output,
    filter_length=WINDOW_SIZE,
    nb_filter=NB_FILTERS,
    activation="relu",
    pooling=True
)
print("convolution layer shape: ", conv_layer_fst.output.shape)

# Dense layer individually
dense_fst_layer = Dense(
    input_data=conv_layer_fst.output,
    input_dim=conv_layer_fst.get_output_dim,
    output_dim=hidden_dim
)
dense_sec_layer = Dense(
    input_data=conv_layer_sec.output,
    input_dim=conv_layer_sec.get_output_dim,
    output_dim=hidden_dim
)

# Merge
pairwise_merge_layer = tf.concat(
    values=[dense_fst_layer.output,
            dense_sec_layer.output],
    axis=-1
)
print(pairwise_merge_layer.shape)

# Dropout
dropout = Dropout(
    input_data=pairwise_merge_layer,
    keep_prob=0.5
)

print("Dropout output_dim: ", int(dropout.output.get_shape()[-1]))
# Dense combine
dense_output_layer = Dense(
    input_data=dropout.output,
    input_dim=int(dropout.output.get_shape()[-1]),
    output_dim=NB_CLASSES,
    activation="relu"
)

loss = dense_output_layer.loss(label_ph) + \
    0.01 * tf.nn.l2_loss(dense_output_layer.weights)
optimizer = tf.train.AdamOptimizer()  # Adam
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)

# init
print("Initialize model...")
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
init = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(init)


def shuffle():
    np.random.seed(1338)
    np.random.shuffle(fst_train_data_sentences)
    np.random.seed(1338)
    np.random.shuffle(fst_train_data_positions)
    np.random.seed(1338)
    np.random.shuffle(sec_train_data_sentences)
    np.random.seed(1338)
    np.random.shuffle(sec_train_data_positions)
    np.random.seed(1338)
    np.random.shuffle(train_labels)


def evaluate(fst_data_sentences, fst_data_positions,
             sec_data_sentences, sec_data_positions,
             data_labels):
    pre_labels = []
    pre_op = dense_output_layer.get_pre_y()
    nb_dev = int(len(data_labels) / BATCH_SIZE)
    for i in range(nb_dev):
        fst_word_feed = fst_data_sentences[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        fst_pos_feed = fst_data_positions[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        sec_word_feed = sec_data_sentences[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        sec_pos_feed = sec_data_positions[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        feed_dict = {
            input_fst_word_ph: fst_word_feed,
            input_fst_pos_ph: fst_pos_feed,
            input_sec_word_ph: sec_word_feed,
            input_sec_pos_ph: sec_pos_feed
        }
        pre_temp = sess.run(pre_op, feed_dict=feed_dict)
        pre_labels += list(pre_temp)

    right_labels = data_labels[: len(pre_labels)]
    acc, pre, f = sim_compute(pre_labels, right_labels)
    return f

# train
print("train model...")
nb_train = int(len(train_labels) / BATCH_SIZE)  # 每次迭代训练次数
for step in tqdm(range(NB_EPOCH)):
    print("Epoch %d: " % step)
    total_loss = 0
    shuffle()
    bound = int(len(train_labels) * 0.75)  # 训练数据进一步划分为训练集和验证集　3:1
    for i in range(nb_train):
        fst_word_feed = fst_train_data_sentences[i *
                                                 BATCH_SIZE: (i + 1) * BATCH_SIZE]
        sec_word_feed = sec_train_data_sentences[i *
                                                 BATCH_SIZE: (i + 1) * BATCH_SIZE]
        fst_pos_feed = fst_train_data_positions[i *
                                                BATCH_SIZE: (i + 1) * BATCH_SIZE]
        sec_pos_feed = sec_train_data_positions[i *
                                                BATCH_SIZE: (i + 1) * BATCH_SIZE]
        label_feed = train_labels[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        feed_dict = {
            input_fst_word_ph: fst_word_feed,
            input_fst_pos_ph: fst_pos_feed,
            input_sec_word_ph: sec_word_feed,
            input_sec_pos_ph: sec_pos_feed,
            label_ph: label_feed
        }
        _, loss_value = sess.run([train_op, loss],
                                 feed_dict=feed_dict)
        total_loss += loss_value

    total_loss = total_loss / float(nb_train)

    # 计算在训练集、测试集上的性能
    f_train = evaluate(fst_train_data_sentences, fst_train_data_positions,
                       sec_train_data_sentences, sec_train_data_positions,
                       train_labels)
    f_test = evaluate(fst_valid_data_sentences, fst_valid_data_positions,
                      sec_valid_data_sentences, sec_valid_data_positions,
                      valid_labels)
    print('\tloss=%f, train f=%f, test f=%f' % (total_loss, f_train, f_test))

print("Done! Time costs: ", float(time() - start_time) / 3600.)
