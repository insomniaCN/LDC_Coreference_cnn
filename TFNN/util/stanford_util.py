#!/usr/bin/env python
# coding=utf-8

from nltk.tag import StanfordPOSTagger

class POSTagger(object):
    """
    调用stanford词性标注工具
    """

    def __init__(self, path_to_jar, model_filename):
        """
        Args:
            path_to_jar: path to 'stanford-postagger.jar'
            model_filename: e.g., chinese-distsim.tagger
        """
        self._path_to_jar = path_to_jar
        self._model_filename = model_filename

        self._postagger = StanfordPOSTagger(
            path_to_jar=self._path_to_jar,
            model_filename=self._model_filename)


    def cn_pos_tag(self, tokens, allow_error=False):
        """
        中文词性标注
        Args:
            tokens: list
        Returns:
            result: tags. e.g,. ['NN', 'VT', 'JJ']
        """
        token_tags = self._postagger.tag(tokens)
        tags = []
        for item in token_tags:
            word_tag = item[1]
            r_index = word_tag.rindex('#')
            tag = word_tag[r_index+1:]
            tags.append(tag)
        if not allow_error:
            if len(tags) != len(tokens):
                return None
        return tags

    def en_pos_tag(self, tokens, allow_error=False):
        """
        英文词性标注
        Args:
            tokens: list
        Returns:
            result: tags. e.g,. ['NN', 'VT', 'JJ']
        """
        token_tags = self._postagger.tag(tokens)
        tags = []
        for item in token_tags:
            word_tag = item[1]
            tags.append(word_tag)
        if not allow_error:
            if len(tags) != len(tokens):
                return None
        return tags

def demo():
    root_path = '/home/zhlu/stanford_tools/stanford-postagger-full-2015-12-09/'
    path_to_jar = root_path + 'stanford-postagger.jar'
    cn_model_filename = root_path + 'models/chinese-distsim.tagger'
    en_model_filename = root_path + 'models/english-bidirectional-distsim.tagger'
    tagger = POSTagger(path_to_jar, en_model_filename)
    #result = tagger.cn_pos_tag("今天 天气 很好!".split(' '))
    result = tagger.en_pos_tag("this is a good day !".split(" "))
    print(result)


if __name__ == '__main__':
    demo()
