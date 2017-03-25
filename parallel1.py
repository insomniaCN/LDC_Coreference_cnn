#!/usr/bin/env python
# coding=utf-8

#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
import sys
sys.path.append("./TFNN/util/")
from stanford_util import POSTagger 
from util import read_all_lines
from dataset import return_voc
from build_corpus import add_pos_tag

add_pos_tag("/home/zhlu/Documents/LDC_Coreference/corpus_handle/test_neg.txt")
