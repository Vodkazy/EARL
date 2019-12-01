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
from ReScorer import *
import json

# encoding=utf8
import sys
import time
reload(sys)
sys.setdefaultencoding('utf8')

f = open('/home/vodka/Desktop/EEARL/log', 'w')

s = ShallowParser()
e = ErPredictor()
t = TextMatch()
j = JointLinker()
r = ReRanker()
z = ReScorer()

def initAllParameters():
    r.change_type_arr = {}
    r.change_flag = False


if __name__ == '__main__':
    # question = str(sys.argv[1])
    ff = open('./data/lcquad.json')
    entity_correct = 0
    entity_wrong = 0
    relation_correct = 0
    relation_wrong = 0
    questions = json.load(ff)
    ff.close()
    # questions = sorted(questions.iteritems(), key=lambda x: x[0], reverse=False)
    for _index, info in enumerate(questions):
        print "Now running the No." + str(_index) + " question..."
        question = info['question']
        f.write(question + "\n")
        initAllParameters()
        r.score_list = {}
        r.change_time = {}
        result_key_chunks = s.shallowParse(question)
        er_predict_result = e.erpredict(result_key_chunks)
        matched_chunks = t.textMatch(er_predict_result)
        topk_res = j.jointLink(matched_chunks)
        final_res = r.reRank(topk_res)
        final_res = z.reScore(final_res,question)
        # Judge whether to rerun the pipeline depending on the flag of finall_res['rejudge']
        if (final_res['rejudge'] == True):
            # print "Rejudge the type of some ontologies..."
            for __index, change_type_flag in final_res['correct-list'].iteritems():
                if change_type_flag == True:
                    if er_predict_result[__index]['class'] == 'entity':
                        er_predict_result[__index]['class'] = 'relation'
                    elif er_predict_result[__index]['class'] == 'relation':
                        er_predict_result[__index]['class'] = 'entity'
            # Init and rerun
            initAllParameters()
            matched_chunks = t.textMatch(er_predict_result)
            topk_res = j.jointLink(matched_chunks)
            final_res = r.reRank(topk_res)
            final_res = z.reScore(final_res, question)
        f.write(str(final_res) + "\n")
#         for entity in info['entity mapping']:
#             flag = False
#             entity_label = entity['label']
#             entity_uri = entity['uri']
#             for _i, item in enumerate(final_res['chunktext']):
#                 if unicode(entity_label) == unicode(item['chunk']):
#                     flag = True
#                     if unicode(entity_uri) == unicode(final_res['rerankedlists'][_i]):
#                         entity_correct += 1
#                         break
#                     else:
#                         entity_wrong += 1
#                         break
#             if flag == True:
#                 continue
#             else:
#                 entity_wrong += 1
#                 continue
#         for relation in info['predicate mapping']:
#             flag = False
#             relation_label = relation['label']
#             relation_uri = relation['uri']
#             for _i, item in enumerate(final_res['chunktext']):
#                 if unicode(relation_label) == unicode(item['chunk']):
#                     flag = True
#                     if unicode(relation_uri) == unicode(final_res['rerankedlists'][_i]):
#                         relation_correct += 1
#                         break
#                     else:
#                         relation_wrong += 1
#                         break
#             if flag == True:
#                 continue
#             else:
#                 relation_wrong += 1
#                 continue
#         print "entity_correct:" + str(entity_correct) + ";entity_wrong:" + str(
#             entity_wrong) + ";relation_correct:" + str(relation_correct) + ";relation_wrong:" + str(
#             relation_wrong) + "\n"
#         f.write("No. " + str(_index) + " question. Entity accuracy is %.6f \n" % (
#         float(entity_correct) / (float(entity_wrong) + float(entity_correct))))
#         f.write("No. " + str(_index) + " question. Relation accuracy is %.6f \n" % (
#         float(relation_correct) / (float(relation_wrong) + float(relation_correct))))
#         f.write("\n")
        f.flush()

    f.close()
