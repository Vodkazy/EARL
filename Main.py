#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
  @ Time     : 19-3-25 下午7:53
  @ Author   : Vodka
  @ File     : Main.py
  @ Software : PyCharm
"""

from ShallowParser import ShallowParser
from ErPredictor import ErPredictor
from TextMatch import TextMatch
from JointLinker import JointLinker
from ReRanker import ReRanker
import sys

s = ShallowParser()
e = ErPredictor()
t = TextMatch()
j = JointLinker()
r = ReRanker()

if __name__ == '__main__':
    question = str(sys.argv[1])
    result_key_chunks = s.shallowParse(question)
    er_predict_result = e.erpredict(result_key_chunks)
    matched_chunks = t.textMatch(er_predict_result)
    topk_res = j.jointLink(matched_chunks)
    final_res = r.reRank(topk_res)
    print final_res