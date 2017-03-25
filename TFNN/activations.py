#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

def get_activation(activation=None):
    """
    获取激活函数

    Args:
        activation: str, 激活函数名称
    Returns:
        激活函数
    """
    if activation is None:
        return None
    elif activation == "tanh":
        return tf.nn.tanh
    elif activation == 'relu':
        return tf.nn.relu
    elif activation == 'softmax':
        return tf.nn.softmax
    elif activation == 'sigmoid':
        return tf.nn.sigmoid
    else:
        raise Exception("Unknow activation fuction: %s " % activation)
