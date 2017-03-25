#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function
from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, Imputer

data = pd.read_csv("./wordsim353/combined.csv")
wordList = np.array(data.iloc[:, [0,1]])  # 一组词
simScore = np.array(data.iloc[:, [2]])  # 相似度

predScoreList = np.zeros((len(simScore), 1))
for i, (word1, word2) in enumerate(wordList):
    print("process #%d words pair [%s, %s]" % (i, word1, word2))
    count = 0
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    for synset1 in synsets1:
        for synset2 in synsets2:
            score = synset1.path_similarity(synset2)
            if score is not None:
                predScoreList[i, 0] += score
                count += 1
            else:
                print(synset1, "path_similarity", synset2, "is None", "=="*10)
    predScoreList[i, 0] = predScoreList[i, 0] * 1.0 / count

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
impList = imp.fit_transform(predScoreList)
mms = MinMaxScaler(feature_range=(0.0, 10.0))
impMmsList = mms.fit_transform(impList)

(coef1, pvalue) = stats.spearmanr(simScore, impMmsList)

submitData = np.hstack((wordList, simScore, impMmsList))
(pd.DataFrame(submitData)).to_csv("wordnet.csv", index=False, header=
    ["word1","word2","OriginSimilarity", "PredSimilarity"])
