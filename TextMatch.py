#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
  @ Time     : 19-3-21 下午3:11
  @ Author   : Vodka
  @ File     : TextMatch.py
  @ Software : PyCharm
"""
import json
import sys
import gensim
import numpy as np
from elasticsearch import Elasticsearch

reload(sys)
sys.setdefaultencoding('utf-8')


class TextMatch:
    def __init__(self):
        print "TextMatch initializing"
        try:
            self.es = Elasticsearch()
            self.labelhash = {}
            self.cache = {}
            file = open('../EARL/data/ontologylabeluridict.json')
            buff_r = file.read()
            self.labelhash = json.loads(buff_r)
            self.model = gensim.models.KeyedVectors.load_word2vec_format(
                '../EARL/data/lexvec.commoncrawl.300d.W.pos.neg3.vectors')
        except Exception, e:
            print e
            sys.exit(1)
        print "TextMatch initialized"

    def pharse_similarity(self, _phrase_1, _phrase_2):
        """
        :param _phrase_1: 
        :param _phrase_2: 
        :return: cos_similarity:
        """
        phrase_1 = _phrase_1.split(" ")
        phrase_2 = _phrase_2.split(" ")
        vector_phrase_1 = []
        vector_phrase_2 = []
        for word in phrase_1:
            try:
                vector_phrase_1.append(self.model.word_vec(word.lower()))
            except:
                continue
        for word in phrase_2:
            try:
                vector_phrase_2.append(self.model.word_vec(word.lower()))
            except:
                continue
        if len(vector_phrase_1) == 0 or len(vector_phrase_2) == 0:
            return 0
        # calculate the average vector of one vector set
        mean_vec_phrase_1 = np.mean(vector_phrase_1, axis=0)
        mean_vec_phrase_2 = np.mean(vector_phrase_2, axis=0)
        cos_similarity = np.dot(mean_vec_phrase_1, mean_vec_phrase_2) / (
            np.linalg.norm(mean_vec_phrase_1) * (np.linalg.norm(mean_vec_phrase_2)))
        return cos_similarity


if __name__ == '__main__':
    t = TextMatch()
    print t.pharse_similarity("the mother", "the father")
    # Result is 0.859742
