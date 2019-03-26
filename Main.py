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


def initAllParameters():
    r.change_type_arr = {}
    r.change_flag = False


if __name__ == '__main__':
    question = str(sys.argv[1])
    result_key_chunks = s.shallowParse(question)
    er_predict_result = e.erpredict(result_key_chunks)
    matched_chunks = t.textMatch(er_predict_result)
    topk_res = j.jointLink(matched_chunks)
    final_res = r.reRank(topk_res)
    # Judge whether to rerun the pipeline depending on the flag of finall_res['rejudge']
    while (final_res['rejudge'] == True):
        print "Rejudge the type of some ontology..."
        for _index, change_type_flag in final_res['correct-list'].iteritems():
            if change_type_flag == True:
                if er_predict_result[_index]['class'] == 'entity':
                    er_predict_result[_index]['class'] = 'relation'
                elif er_predict_result[_index]['class'] == 'relation':
                    er_predict_result[_index]['class'] = 'entity'
        # Init and rerun
        initAllParameters()
        matched_chunks = t.textMatch(er_predict_result)
        topk_res = j.jointLink(matched_chunks)
        final_res = r.reRank(topk_res)
    print final_res
#