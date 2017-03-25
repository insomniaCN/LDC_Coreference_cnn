# -*- coding: utf-8 -*-
"""
    command line:
        
        python3 train_word2vec_model.py path_of_corpus path_of_model
        
        python3 train_word2vec_model.py ./data_handle/car_data.txt ./model/xx_model.bin
"""
import logging
import os.path
import sys
import multiprocessing
from gensim.models.word2vec import LineSentence
from time import time
from gensim.models import Word2Vec
from util import read_lines

if __name__ == '__main__':
    t0 = time()

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
 
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
 
    # check and process input arguments
    if len(sys.argv) < 3:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    inp, outp = sys.argv[1:3]  # corpus path and path to save model
 
    model = Word2Vec(sg=1, sentences=LineSentence(inp), size=90, window=5, min_count=3,
            workers=16, iter=40)
 
    # trim unneeded model memory = use(much) less RAM
    #model.init_sims(replace=True)
    model.save_word2vec_format(outp, binary=True)

    print('done in %ds!' % (time()-t0))

