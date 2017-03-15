#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import time
import numpy as np
import tensorflow as tf
from dataset import load_data_nostatic
from Layers import Embedding, Convolutional1D, \
					MaxPooling1D, Dense
from evaluate import sim_compute, compute


# 参数设置
BATCH_SIZE = 50
NB_EPOCH = 50
WINDOW_SIZE = 3
NB_FILTERS = 150
WORD_EMBED_DIM = 300
POS_EMBED_DIM = 50
NB_CLASSES = 2


def evaluate(data_sentences, data_positions, data_labels):
	pre_labels = []
	pre_op = dense_layer.get_pre_y()
	nb_dev = int(len(data_labels) / BATCH_SIZE)
	for i in range(nb_dev):
		word_feed = data_sentences[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
		pos_feed = data_positions[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
		feed_dict = {
			input_word_ph: word_feed,
			input_pos_ph: pos_feed
		}
		pre_temp = sess.run(pre_op, feed_dict=feed_dict)
		pre_labels += list(pre_temp)

	right_labels = data_labels[: len(pre_labels)]
	acc, pre, f = sim_compute(pre_labels, right_labels, ignore_labels=8)


# t0 = time()
# 加载数据，训练：测试 = 97876:58832
print("loading data...")
bound = 97000  # 训练数据数量
fst_data_sentences, sec_data_setences, fst_data_positions, sec_data_positions, \
    labels, word_embed_weights, pos_embed_weights, type_voc, max_len = \
	load_data_nostatic(pos_embed_dim=POS_EMBED_DIM)
train_data_sentences, train_data_positions, train_labels = \
	data_sentences[:bound], data_positions[:bound], labels[:bound]
test_data_sentences, test_data_positions, test_labels = \
	data_sentences[bound:], data_positions[bound:], labels[bound:]
print("data_senteces shape:", data_sentences.shape)
print("data_positions shape:", data_positions.shape)
print("labels shape:", labels.shape)
# 构建模型
print("build model...")
input_word_ph = tf.placeholder(
	tf.int32,
	shape=(None, max_len*2)
)
input_pos_ph = tf.placeholder(
	tf.int32,
	shape=(None, max_len*2)
)
label_ph = tf.placeholder(
	tf.int32,
	shape=(None)
)
word_embed_weights = tf.Variable(word_embed_weights, tf.float32)
word_embed_layer = Embedding(  # word embedding
	params=word_embed_weights,
	ids=input_word_ph
)
pos_embed_weights = tf.Variable(pos_embed_weights, dtype=tf.float32)
pos_embed_layer = Embedding(
	params=pos_embed_weights,
	ids=input_pos_ph
)
pos_embed_output = tf.reshape(  # reshape, 为了将词和位置拼接
	pos_embed_layer.output,
	[BATCH_SIZE, max_len*2, 2*POS_EMBED_DIM]
)
word_pos_output = tf.concat(  # 拼接词和位置
	values=[word_embed_layer.output, pos_embed_output],
        axis=2
)
conv_layer_2 = Convolutional1D(
	input_data=word_pos_output,
	filter_length=WINDOW_SIZE,
	nb_filters=NB_FILTERS,
	activation='relu'
)
pool_layer_2 = MaxPooling1D(
	input_data=conv_layer_2.output
)
dense_layer = Dense(
	input_data=pool_layer_2.output,
	input_dim=pool_layer_2.get_output_dim(),
	output_dim=NB_CLASSES
)
dropout = tf.layers.dropout(inputs=dense_layer.output, rate=0.5)

logits = tf.layers.dense(inputs=dropout, units=10)
# loss = dense_layer.loss(label_ph)# + 0.1*tf.nn.l2_loss(dense_layer.weights)
one_hot_labels = tf.one_hot(indices=tf.cast(train_labels, tf.int32),depth=10)
loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
optimizer = tf.train.AdamOptimizer()  # Adam
global_step = tf.Variable(0, name='global_step', trainabel=False)
train_op = optimizer.minimize(loss, global_step=global_step)

# init
print("Initialize model...")
init = tf.initialize_all_tables()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(init)


def shuffle():
	np.random.seed(1338)
	np.random.shuffle(train_data_sentences)
	np.random.seed(1338)
	np.random.shuffle(train_data_positions)
	np.random.seed(1338)
	np.random.shuffle(train_labels)


# train
print("train model...")
nb_train = int(len(train_labels) / BATCH_SIZE)  # 每次迭代训练次数
for step in range(NB_EPOCH):
	print("Epoch %d: " % step)
	total_loss = 0
	shuffle()
	bound = int(len(train_labels) * 0.75)  # 训练数据进一步划分为训练集和验证集　3:1
	for i in range(nb_train):
		word_feed = train_data_sentences[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
		pos_feed = train_data_positions[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
		label_feed = train_labels[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
		feed_dict = {
			input_word_ph: word_feed,
			input_pos_ph: pos_feed,
			label_ph: label_feed
		}
		_, loss_value = sess.run([train_op, loss],
							feed_dict=feed_dict)
		total_loss += loss_value
	total_loss = total_loss / float(nb_train)

	# 计算在训练集、测试集上的性能
	f_train = evaluate(train_data_sentences, train_data_positions, train_labels)
	f_test = evaluate(test_data_sentences, test_data_positions, test_labels)
	print('\tloss=%f, train f=%f, test f=%f' % (total_loss, f_train, f_test))
