# -*- encoding:utf-8 -*-
"""
DNN Layers.

@author: ljx
"""
import math
import numpy as np
import tensorflow as tf

def get_activation(activation=None):
    """
    Get activation function accord to the parameter 'activation'
    :param activation: str: 激活函数的名称
    :return: 激活函数
    """
    if activation is None:
        return None
    elif activation == 'tanh':
        return tf.nn.tanh
    elif activation == 'relu':
        return tf.nn.relu
    else:
        raise Exception('Unknow activation function: %s' % activation)


class Convolutional1D(object):

    def __init__(self, input_data, filter_length, nb_filters, strides=[1,1,1,1],
                 padding='VALID', pool_length=None, activation=None,
                 weights=None, biases=None, name='Convolutional1D'):
        """1D卷积层
        input_data: 3D tensor of shape=[batch_size, in_height, in_width]
            in_channels is set to 1 when use Convolutional1D.
        filter_length: int, 卷积核的长度，用于构造卷积核，在Convolutional1D中，
            卷积核shape=[filter_length, in_width, in_channels, nb_filters]
        nb_filters: int, 卷积核数量
        padding: 默认'VALID'，暂时不支持设成'SAME'
        weights: np.ndarray, 卷积核权重
        biases: np.ndarray, bias
        """
        assert padding in ('VALID'), 'Unknow padding %s' % padding
        # assert padding in ('VALID', 'SAME'), 'Unknow padding %s' % padding

        in_height, in_width = map(int, input_data.get_shape()[1:])
        self._input_data = tf.expand_dims(input_data, -1)  # shape=[x, x, x, 1]
        self._filter_length = filter_length
        self._nb_filters = nb_filters
        self._strides = strides
        self._padding = padding
        self._activation = get_activation(activation)

        # 构造卷积核权重
        fan_in = 1 * filter_length * in_width
        fan_out = nb_filters * filter_length * in_width / float(in_height)
        w_bound = tf.sqrt(6. / (fan_in + fan_out))
        if weights is None:
            weights = tf.Variable(
                tf.random_uniform(
                    shape = [filter_length, in_width, 1, nb_filters],
                    minval = -w_bound,
                    maxval = w_bound
                )
            )
        self._weights = weights

        # bias
        if biases is None:
           biases = tf.zeros(nb_filters)
        self._biases = biases

        # 卷积  if padding='VALID', then conv_output's shape=
        #   [batch_size, in_height-filter_length+1, 1, nb_filters]
        conv_output = tf.nn.conv2d(
            input = self._input_data,
            filter = self._weights,
            strides = self._strides,
            padding = self._padding)

        # output's shape=[batch_size, new_height, 1, nb_filters]
        linear_output = tf.nn.bias_add(conv_output, self._biases)
        self._output = (
            linear_output if activation is None
            else self._activation(linear_output))

    @property
    def input_data(self):
        return self._input_data

    @property
    def filter_length(self):
        return self._filter_length

    @property
    def nb_filter(self):
        return self._nb_filters

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    @property
    def output(self):
        return self._output


class MaxPooling1D(object):

    def __init__(self, input_data, strides=[1,1,1,1], padding='VALID'):
        """池化
        input_data: A 4D tensor of shape=
            [batch_size, height, 1, nb_filters]
        """
        self._input_data = input_data
        self._strides = strides
        self._padding = padding

        self._ksize = int(self._input_data.get_shape()[1])
        # pool_output's shape=[batch_size, 1, 1, nb_filters]
        self._pool_output = tf.nn.max_pool(
            value = self._input_data,
            ksize = [1, self._ksize, 1, 1],
            strides = self._strides,
            padding = self._padding)
        # reshape 为 shape=[batch_size, nb_filters]
        self._output = tf.squeeze(self._pool_output, [1,2])

    @property
    def output(self):
        return self._output

    def get_output_dim(self):
        """
        maxpooling 层的输出维度    
        """
        return int(self._output.get_shape()[-1])


class PieceWiseConvolutional1D(object):

    def __init__(self, input_data, filter_length, nb_filters, strides=[1,1,1,1],
                 pool_length=None, activation=None, weights=None, biases=None,
                 name='Convolutional1D'):
        """Piecewise Convolutional 1D Layer.
        input_data: 3D tensor of shape=[batch_size, in_height, in_width]
            in_channels is set to 1 when use Convolutional1D.
        filter_length: int, 卷积核的长度，用于构造卷积核，在Convolutional1D中，
            卷积核shape=[filter_length, in_width, in_channels, nb_filters]
        nb_filters: int, 卷积核数量
        weights: np.ndarray, 卷积核权重
        biases: np.ndarray, bias
        """
        batch_size, in_height, in_width = map(int, input_data.get_shape()[:]) 
        self._input_data = tf.expand_dims(input_data, -1)  # shape[x, x, x, 1]
        self._filter_length = filter_length
        self._nb_filters = nb_filters
        self._strides = strides
        self._activation = get_activation(activation)

        # 构造卷积核权重
        fan_in = 1 * filter_length * in_width
        fan_out = nb_filters * filter_length * in_width / float(in_height/3.)
        w_bound = tf.sqrt(6. / (fan_in + fan_out))
        if weights is None:
            weights = tf.Variable(
                tf.random_uniform(
                    shape = [filter_length, in_width, 1, nb_filters],
                    minval = -w_bound,
                    maxval = w_bound
                )
            )
        self._weights = weights

        # bias
        if biases is None:
           biases = tf.zeros(nb_filters)
        self._biases = biases

        # 卷积  if padding='VALID', then conv_output's shape=
        #   [batch_size, in_height-filter_length+1, 1, nb_filters]
        conv_output = tf.nn.conv2d(
            input = self._input_data,
            filter = self._weights,
            strides = self._strides,
            padding = 'VALID'  # 等长卷积
        )

        # self._output's shape=[batch_size, height, 1, nb_filters]
        linear_output = tf.nn.bias_add(conv_output, self._biases)
        output = (  # shape=[batch_size, height, in_width, nb_filters]
            linear_output if activation is None
            else self._activation(linear_output)
        )
        # shape=[batch_size, height, 1, nb_filters]
        self._output = output

    @property
    def input_data(self):
        return self._input_data

    @property
    def filter_length(self):
        return self._filter_length

    @property
    def nb_filter(self):
        return self._nb_filters

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    @property
    def output(self):
        return self._output


class PieceWiseMaxPooling1D(object):

    def __init__(self, input_data, input_indexs, strides=[1,1,1,1],
                 padding='VALID'):
        """分段池化Layer.
        input_data: A 4D tensor of shape=
            [batch_size, height, 1, nb_filters]
        input_index: A 2D tensor of shape=
            [batch_size, 2], 每个实例中两个实体在句子中的下标
        """
        self._batch_size = int(input_data.get_shape()[0])  # batch size
        self._input_data = tf.squeeze(input_data, [2])  # [bs,height,nb_filters]
        #self._input_data = input_data  # shape=[bs, height, nb_filtersi]
        self._input_indexs = input_indexs
        self._strides = strides
        self._padding = padding
        self._nb_filters = int(input_data.get_shape()[-1])

        output = []  # 池化层输出, shape=[batch_size, nb_filters]
        f_map_height = int(self._input_data.get_shape()[1])
        for i in range(self._batch_size):  # 处理一个batch中的每一个实例
            # 每一个feature map, shape=[1, seg_height, nb_filter], # 两个实体的下标
            index_1, index_2 = self._input_indexs[i, 0],self._input_indexs[i, 1]
            feature_map_seg_1 = tf.slice(self._input_data, [i, 0, 0], 
                                         [1, index_1+1, -1])
            feature_map_seg_2 = tf.slice(self._input_data, [i, index_1, 0],
                                         [1, index_2-index_1+1, -1])
            feature_map_seg_3 = tf.slice(self._input_data, [i, index_2, 0],
                                         [1, -1, -1])
            seg_pool_1 = self.piecewise_default_maxpooling1d(feature_map_seg_1)
            seg_pool_2 = self.piecewise_default_maxpooling1d(feature_map_seg_2)
            seg_pool_3 = self.piecewise_default_maxpooling1d(feature_map_seg_3)

            # 将分段后的拼接起来, seg_pool_*'s shape=[1, nb_filters]
            # 拼接后shape=[3, nb_filters]
            cat_pool = tf.concat(0, [seg_pool_1, seg_pool_2, seg_pool_3])
            # reshape 为[nb_filters, 3]
            cat_pool = tf.transpose(cat_pool, perm=[1, 0])  # sp=[3,nb_filters]
            cat_pool = tf.reshape(cat_pool, [-1])  # shape=[3*nb_filters,]
            cat_pool = tf.expand_dims(cat_pool, [0])  # shape=[1, 3*nb_filters]
            # 将该实例池化得到的结果cat_pool放至output中
            output.append(cat_pool)

        # 将output列表中中batch_size个元素拼接起来
        self._output = tf.concat(0, output)  # shape=[batch_size, 3*nb_filters]

    @property
    def output(self):
        return self._output

    def get_output_dim(self):
        """
        maxpooling 层的输出维度    
        """
        return 3 * self._nb_filters

    def piecewise_default_maxpooling1d(self, input_data):
        """
        使用默认值的piecewise pooling
        input_data's shape=[1, seg_height, nb_filters]
        """
        # reshape 为 [seg_height, nb_filters]
        input_data = tf.squeeze(input_data, [0])
        # pool后shape=[nb_filters,]
        pool_output = tf.reduce_max(input_data, reduction_indices=[0])
        # reshape 为[1, nb_filters]
        seg_pool_output = tf.expand_dims(pool_output, [0])
        # seg_pool_output = tf.reshape(pool_output, [1, 222])
        return seg_pool_output  # shape=[1, nb_filters]


class Embedding(object):

    def __init__(self, params, ids, name='Embedding'):
        self._params = params
        self._ids = ids 

        #output
        self._output = tf.nn.embedding_lookup(
            params = self._params,
            ids = self._ids
        )

    @property
    def params(self):
        return self._params

    @property
    def output(self):
        return self._output


class Dense(object):

    def __init__(self, input_data, input_dim, output_dim, weights=None, 
                 biases=None, activation=None, dropout=None,
                 name='Convolution1D'):
        assert len(input_data.get_shape())==2, \
            "全连接层的输入必须要flatten, 即shape=[batch_size, input_dim]"
        self._input_data = input_data
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._activation = get_activation(activation)
        self._dropout = dropout
        self._name = name

        # initialize weights
        if weights is None:
            w_bound = tf.sqrt(6. / (input_dim + output_dim))
            weights = tf.Variable(
                #tf.truncated_normal(
                #    shape=[input_dim, output_dim],
                #    stddev=1.0/math.sqrt(float(self._input_dim))  # TODO
                #),
                tf.random_uniform(
                    shape = [input_dim, output_dim],
                    minval = -w_bound,
                    maxval = w_bound
                ),
                name='weights'
            )
        self._weights = weights

        # initialize biases
        if biases is None:
            biases = tf.Variable(tf.zeros([self._output_dim]),
                                 name='biases')
        self._biases = biases

        # output
        linear_output = tf.matmul(self._input_data, self._weights) + \
                            self._biases
        self._output = (
            linear_output if self._activation is None
            else self._activation(linear_output)
        )
        if self._dropout:
            self._drop_output = tf.nn.dropout(self._output, \
                                              1-self._dropout)

    def loss(self, y):
        # output_shape = tuple(map(int, self._output.get_shape()))
        # assert output_shape==y.shape, \
        #     'output(shape=%s) does not match with y(shape=%s)' % \
        #         (self._output.get_shape(), y.shape)
        # TODO 写成工具函数，包括交叉熵, etc.
        y = tf.cast(y, tf.int32)
        cross_entroy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            self.output, y, name='xentroy')
        # output = tf.reshape(self._output, [-1])
        # cross_entroy = tf.nn.sigmoid_cross_entropy_with_logits(
        #     output, y, name='xentroy')
        loss = tf.reduce_mean(cross_entroy, name='xentroy_mean')
        return loss

    def get_pre_y(self):
        # TODO 待修改
        # pre_y = tf.reshape(tf.round(tf.sigmoid(self._output)), [-1])
        pre_y = tf.arg_max(input=self._output, dimension=1)
        return pre_y
    
    @property
    def input_data(self):
        return self._input_data

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def name(self):
        return self._name

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases

    @property
    def output(self):
        if not self._dropout:
            return self._output
        else:
            return self._drop_output

