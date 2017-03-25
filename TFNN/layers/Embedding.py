#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

class Embedding(object):
    '''
    embdding层

    Args:
        params: a list tensors with the same shape and type
        ids: a Tensor with type int32　containning the ids to be looked up in params
    Returns:
        output: Tensors shape [None,max_len,embed dim]
    '''

    def __init__(self, params, ids, keep_prob=1.0, name='Embedding'):
        self._params = params
        self._ids = ids
        
        #output
        embed_output = tf.nn.embedding_lookup(
            params = self.params, 
            ids = self.ids  
            )
        self._output = tf.nn.dropout(embed_output, keep_prob)


    @property
    def params(self):
        return self._params


    @property
    def ids(self):
        return self._ids


    @property
    def output(self):
        return self._output
