#-*- encoding:utf-8 -*-
"""
@author: ljx
"""
import re
import codecs

def read_all_lines(path):
    all_lines = []
    with codecs.open(path, 'r', encoding='utf-8') as file:
        temp_lines = file.readlines()
        for line in temp_lines:
            line = line.strip()
            if line:
                all_lines.append(line)
    return all_lines

