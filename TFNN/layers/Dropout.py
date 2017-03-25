#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

class Dropout(object):
    """
    Dropout层
    """

    def __init__(self, input_data, keep_prob=1.0, noise_shape=None, seed=None, 
                 name='Dropout'):
        self._input_data = input_data
        self._keep_prob = keep_prob
        self._noise_shape = noise_shape
        self._seed = seed
        self._name = name
        
        self.call()


    def call(self):
        # output
        self._output = tf.nn.dropout(x=self._input_data,
                                     keep_prob=self._keep_prob,
                                     name='Dropout')


    @property
    def input_data(self):
        return self._input_data


    @property
    def keep_prob(self):
        return self._keep_prob


    @property
    def output(self):
        return self._output
