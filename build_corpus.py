#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from util import read_all_lines
import os
import re
from lxml import etree
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

CORPUS_PATH = './LDC_corpus/data/2015/'
TRAIN_PATH = './LDC_corpus/data/2015/training/event_hopper/'
CORPUS_HANDLE_PATH = "./LDC_corpus/corpus_handle/"
# SOURCE_PATH = './LDC_corpus/data/2015/training/source/'   #training corpus source
SOURCE_PATH = './LDC_corpus/data/2015/eval/source/'
EVAL_PATH = './LDC_corpus/data/2015/eval/hopper/'


def XML_EventHopper_parse(xml_file):
    '''
    extract event hoppers from LDC corpus

    Args:
        xml_file: absolute path of XML files
    Returns:
        event_hopper_clusters: the list of event hoppers with attributes
        doc_id: which doc event hoppers belong to
    '''
    event_hopper_clusters = []
    one_hopper = []

    tree = ET.ElementTree(file=xml_file)  # 打开XML文档
    # parser = etree.XMLParser(recover=True)
    # tree = etree.iterparse(xml_file, parser)
    root = tree.getroot()
    root_tag, root_attrib = root.tag, root.attrib
    doc_id = root_attrib['doc_id']

    for elem in tree.iter():
        # print(elem.tag, elem.attrib)
        if elem.tag == 'deft_ere_event_nuggets' or elem.tag == 'hoppers':
            continue

        if elem.tag == 'hopper':    # 每一个hopper组成一个list
            if one_hopper != []:
                event_hopper_clusters.append(one_hopper)

            one_hopper = []
            one_hopper.append(elem.attrib)
        else:
            one_hopper.append(elem.attrib)  # 加入attrtibutes
            if elem.tag == 'trigger':   # 添加trigger words
                one_hopper.append(elem.text)

    event_hopper_clusters.append(one_hopper)
    del(one_hopper)
    return event_hopper_clusters, doc_id


def XML_EventNugget_parse(xml_file):
    '''
    extract event nuggets from LDC corpus

    Args:
        xml_file: absolute path of XML files
    Returns:
        event_nugget_clusters: the list of event hoppers with attributes
        doc_id: which doc event hoppers belong to
    '''
    event_nugget_clusters = []
    one_nugget = []

    tree = ET.ElementTree(file=xml_file)  # 打开XML文档
    root = tree.getroot()
    root_tag, root_attrib = root.tag, root.attrib
    doc_id = root_attrib['doc_id']

    for elem in tree.iter():
        # print(elem.tag, elem.attrib)
        if elem.tag == 'deft_ere_event_nuggets_only':
            continue

        if elem.tag == 'event_mention':    # 每一个hopper组成一个list
            if one_nugget != []:
                event_nugget_clusters.append(one_nugget)

            one_nugget = []
            one_nugget.append(elem.attrib)
        else:
            one_nugget.append(elem.attrib)  # 加入attrtibutes
            one_hopper.append(elem.text)

    del(one_nugget)
    return event_nugget_clusters, doc_id


# def textWithString(source_file):
#     """
#     将文本保存到一个string中，并给出每段的长度
# 
#     Args：
#         source_file: 原语料位置
#     Returns:
#         all_lines:所有段落组成的list
#         paragraph_length: 每段最后字的位置
# 
#     """
#     text_str = ''   # 保存整个文本
#     paragraph_length = [-1]
#     all_lines = read_all_lines(source_file)
#     # line_one = open(source_file).read()
#     # print(line_one.find('says'))
#     lines_parag = []  # 将文本按段落保存
#     for line in all_lines:
#         if line.startswith('<') and line.endswith('>'):  # 每个XML标签自成一格段落
#             lines_parag.append(line+'\n')
#             lines_parag.append('')
#             continue
#         if line == '':  # 对空行加一格占位符
#             lines_parag[-1] += '\n'
#             lines_parag.append('')
#             continue
# 
#         if not line.endswith('.'):
#             lines_parag[-1] += line+'\n'
#             lines_parag.append('')
#         else:
#             lines_parag[-1] += line+'\n'
# 
#     if lines_parag[-1] == '':
#         del lines_parag[-1]
#     del all_lines
# 
#     for line in lines_parag:
#         # print(line+'##')
#         text_str += line+'\n'
#         paragraph_length.append(len(line)+paragraph_length[-1])
#     # paragraph_length.append(999999) #检测trigger是否在最后一段
# 
#     return lines_parag, paragraph_length
# 
# 
def textWithString(source_file):
    """
    去除xml标签、空行、换行符等并将剩余文本按句子的形式存于列表中

    Args:
        source_file：source语料位置
    Returns:
        sentence：保存为列表的句子集合
        text_str：source原文本字符串
    """
    text_str = open(source_file, 'r').read()
    text_split = re.split('<.*>', text_str)
    text_split = list(map(lambda x: x.strip().strip('\n'), text_split))
    text_split = list(filter(lambda x: x != '', text_split))
    text_split = list(map(lambda x: x.replace('\n', ' '), text_split))  # 去除xml标签，空行，换行符
    # print(text_split)

    sentence = []
    for text in text_split:
        sentence.extend(re.split('\.\.\.|\.|\?', text))
    sentence = list(filter(lambda x: x != '', sentence))  # 将文本按句子划分

    return sentence, text_str


def events2str(event, doc_id):
    """
     将event由list转化为string，取其中一部分的attributes.

    Args:
        event: event list
        doc_id: 文本id
    Returns:
        event_str: event string
    """
    source_file = SOURCE_PATH + doc_id + '.txt'
    sentence, text_str = textWithString(source_file)
    pattern = False
    selected_sen = ''
    for sen in sentence:  # 分别向后或者向前查找该词，若存在则记录下该词所在的句子
        if text_str[int(event[1]['offset']): int(event[1]['offset'])+12] in sen:
            selected_sen = sen
            pattern = True
        if not pattern:  # 若向后找不着则在向前查找
            if text_str[int(event[1]['offset'])-12: int(event[1]['offset'])] in sen:
                selected_sen = sen
                pattern = True

    if pattern:  # 找到对应的句子则返回相应的信息，没有找到则跳过该词对
        sentence_split = re.split(r'\W', selected_sen)
        relative_pos = sentence_split.index(event[2].split(' ')[0])  # 词在句子中的相对位置

        # event hopper id | event type | event subtype | event realis | relative position |
        # current sentence | event mention 
        event_str = event[0]['id'] + '|' + event[0]['type'] + '|' + event[0]['subtype'] + '|' + \
            event[0]['realis'] + '|' + str(relative_pos) + '|' + \
            selected_sen.strip('\n') + '|' + event[2] + '\n'

        return event_str
    else:
        print("current events can't be recognized correctly! ")
        event_str = ''
        return event_str


# def events2str(event, doc_id):
#     """
#     将event由list转化为string，取其中一部分的attributes.
# 
#     Args:
#         event: event list
#         doc_id: 文本id
#     Returns:
#         event_str: event string
#     """
#     source_file = SOURCE_PATH + doc_id + '.txt'
#     current_paragraph = ''  # 事件所在的段落
# 
#     all_lines, paragraph_length = textWithString(source_file)
#     # print(paragraph_length)
#     for i in range(len(paragraph_length)-2):
# 
#         if int(event[1]['offset']) > paragraph_length[i] and int(event[1]['offset']) <= paragraph_length[i+1]:
#             current_paragraph = all_lines[i]
#             # print(all_lines[i])
#             break
# 
#     parag_split = re.split(r'\W', current_paragraph)
#     # print(parag_split)
#     # parag_split = current_paragraph.strip().split(' ')    #分词
#     # print(parag_split, event[2])
#     relative_pos = parag_split.index(event[2])  # 词在句子中的相对位置
# 
#     # event id|event type|event subtype|event realis|trigger relative position|sentence|trigger word
#     event_str = event[0]['id'] + '|' + event[0]['type'] + '|' + event[0]['subtype'] + '|' + \
#         event[0]['realis'] + '|' + str(relative_pos) + '|' + \
#         current_paragraph.strip('\n')+ '|' + event[2] + '\n'
# 
#     return event_str
# 
def build_corpus(event_hopper_clusters, doc_id):
    """
    将语料处理为event-pair形式，一个pair在文本中占两行

    Args:
        event_hopper_clusters: the list of event hoppers with attributes
        doc_id: which doc event hoppers belong to
    Returns:
        None
    """

    # train_pos = CORPUS_HANDLE_PATH+'train_pos.txt'
    # train_neg = CORPUS_HANDLE_PATH+'train_neg.txt'
    train_pos = CORPUS_HANDLE_PATH+'test_pos.txt'
    train_neg = CORPUS_HANDLE_PATH+'test_neg.txt'

    if not os.path.exists(train_pos):
        f = open(train_pos, 'w')
        f.close()
    if not os.path.exists(train_neg):
        f = open(train_neg, 'w')
        f.close()

    f_pos = open(train_pos, 'a')
    f_neg = open(train_neg, 'a')
    if event_hopper_clusters == []:
        print("doc_id: %s has no hoppers!" % doc_id)

    for event_hopper_cluster in event_hopper_clusters:
        try:
            # print(event_hopper_cluster)
            events = []
            # events_num = len(event_hopper_cluster) / 3
            for i in range(1, len(event_hopper_cluster), 3):
                events.append(event_hopper_cluster[i:i+3])  # 每个事件组成一个list,事件集组成一个list
            # print(events)
            for i in range(0, len(events)-1):

                # print(events[i])
                event_strI = events2str(events[i], doc_id)
                if event_strI == '':
                    continue
                for j in range(i+1, len(events)):

                    event_strJ = events2str(events[j], doc_id)
                    if event_strJ == '':
                        continue
                    if event_strI.split('|')[5] == event_strJ.split('|')[5]:
                        # 对完全一样的句子不做处理
                        continue
                    f_pos.write(event_strI+event_strJ)  # pos_examples

            intra_events = []   # 其余hoppers中所有的events
            for event_hopper_cluster2 in event_hopper_clusters:
                if event_hopper_clusters.index(event_hopper_cluster2) <= \
                event_hopper_clusters.index(event_hopper_cluster):  # 避免重复处理和hopper间处理
                    continue

                each_events = []
                for i in range(1, len(event_hopper_cluster2), 3):
                    each_events.append(event_hopper_cluster2[i:i+3])
                intra_events.extend(each_events)

            for e1 in events:
                e1_str = events2str(e1, doc_id)
                if e1_str == '':  # 对未能正确找到所在句子的事件做跳过处理
                    continue
                for e2 in intra_events:
                    e2_str = events2str(e2, doc_id)
                    if e2_str == '':
                        continue
                    f_neg.write(e1_str+e2_str)  # neg_examples
        except ValueError:
            print("event has been skipped..")

    f_pos.close()
    f_neg.close()


def build_train_corpus():
    """
    构建测试语料

    Args:
        None
    Returns:
        None
    """
    for root, dirs, files in os.walk(TRAIN_PATH):
        for fl in files:
           #  try:
           #      filepath = os.path.join(root, fl)
           #      event_hopper_clusters, doc_id = XML_EventHopper_parse(filepath)
           #      build_corpus(event_hopper_clusters, doc_id)
           #      print("file %s has been done." % fl)
           #  except ValueError:
           #      print("file %s hasn't been done" % fl)
            filepath = os.path.join(root, fl)
            event_hopper_clusters, doc_id = XML_EventHopper_parse(filepath)
            build_corpus(event_hopper_clusters, doc_id)


def build_test_corpus():
    """
    构建测试语料
    """
    for root, dirs, files in os.walk(EVAL_PATH):
        for fl in files:
            filepath = os.path.join(root, fl)
            event_hopper_clusters, doc_id = XML_EventHopper_parse(filepath)
            build_corpus(event_hopper_clusters, doc_id)


if __name__ == '__main__':
    # build_train_corpus()
    build_test_corpus()
    print("Congrats! Done! ")
