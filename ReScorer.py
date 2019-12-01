#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
  @ Time     : 19-4-16 下午9:28
  @ Author   : Vodka
  @ File     : ReScorer.py
  @ Software : PyCharm Community Edition
"""
import sys

import numpy as np
import torch
from script.Features import Features
from script.Transform2vec import Transform2vec
from script.ComparePredictor import Rnn

reload(sys)
sys.setdefaultencoding('utf8')


class ReScorer:
    def __init__(self):
        self.j = Transform2vec()
        self.f = Features()
        self.c = torch.load('./model/compare_predictor.model')
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    def reScore(self, rerank_result, question):
        """
        Use the context and uri properties to rescore the grade for each uri
        :param rerank_result:
        :param question:
        :return:
        """
        rescore_result = rerank_result
        for _index, uris in rerank_result['rerankedlists'].iteritems():
            if rerank_result['types'][_index] == 'entity':
                rescore_result['rerankedlists'][_index] = \
                sorted(rescore_result['rerankedlists'][_index], key=lambda x: x[0], reverse=True)[0][1]
                continue
            label = rerank_result['chunktext'][_index]['chunk']
            context_vec = self.j.transContext2Vecs(question, label)
            cX = context_vec[0]
            for __index, item in enumerate(context_vec):
                if __index != 0:
                    cX = cX + item
            cX = torch.from_numpy(cX.astype('float32'))
            cX = cX.reshape(1, 1, len(cX))
            context_vec = self.c(cX)
            for ___index, uri in enumerate(uris):
                # uri[0] is score,uri[1] is uri
                _domain = self.f.getDomainVec(uri[1])
                _range = self.f.getRangeVec(uri[1])
                _type = self.f.getTypeVec(uri[1])
                _uri = self.f.getUriVec(uri[1])
                property_vec = np.hstack((_domain, _range))
                property_vec = np.hstack((property_vec, _type))
                property_vec = np.hstack((property_vec, _uri))
                p = []
                p.append(property_vec)
                p = np.array(p)
                property_vec = torch.from_numpy(p)
                score_connect = uri[0]
                score_context = self.cos(context_vec, property_vec)
                score = score_connect * 0.6 + score_context * 0.4
                rescore_result['rerankedlists'][_index][___index] = (score, uri[1])
            rescore_result['rerankedlists'][_index] = \
            sorted(rescore_result['rerankedlists'][_index], key=lambda x: x[0], reverse=True)[0][1]
            # print rescore_result
        return rescore_result

# if __name__ == '__main__':
#     r = ReScorer()
#     r.reScore({'rerankedlists': {1: [(0.25071218609809875, u'http://dbpedia.org/resource/Family_of_Barack_Obama'),
#                                      (0.1938347965478897, u'http://dbpedia.org/resource/Barack_Obama,_Sr.'),
#                                      (0.007336166687309742,
#                                       u'http://dbpedia.org/resource/Barack_Obama_%22Hope%22_poster'),
#                                      (0.004728458821773529,
#                                       u'http://dbpedia.org/resource/Barack_Obama_%22Joker%22_poster'),
#                                      (0.0014046295545995235, u'http://dbpedia.org/resource/Barack_Obama'),
#                                      (0.0011621536687016487, u'http://dbpedia.org/resource/Presidency_of_Barack_Obama'),
#                                      (0.0004659400146920234,
#                                       u'http://dbpedia.org/resource/The_Case_Against_Barack_Obama'), (
#                                          0.000256964034633711,
#                                          u"http://dbpedia.org/resource/Confirmations_of_Barack_Obama's_Cabinet"),
#                                      (0.0002468969614710659, u'http://dbpedia.org/resource/Barack_Obama:_The_Story'),
#                                      (0.00024368573212996125, u'http://dbpedia.org/resource/Speeches_of_Barack_Obama'),
#                                      (0.0002059761609416455,
#                                       u'http://dbpedia.org/resource/Barack_Obama_on_social_media'), (
#                                          0.00017089983157347888,
#                                          u'http://dbpedia.org/resource/By_the_People:_The_Election_of_Barack_Obama'),
#                                      (0.00015553012781310827,
#                                       u'http://dbpedia.org/resource/Public_image_of_Barack_Obama'),
#                                      (0.00015441638242918998, u'http://dbpedia.org/resource/Barack_Obama_Academy'),
#                                      (0.0001491083821747452,
#                                       u'http://dbpedia.org/resource/Inauguration_of_Barack_Obama'),
#                                      (0.00012262043310329318, u'http://dbpedia.org/resource/Barack_Obama_in_comics'), (
#                                          0.00011838223872473463,
#                                          u"http://dbpedia.org/resource/The_Speech:_Race_and_Barack_Obama's_%22A_More_Perfect_Union%22"),
#                                      (0.00011634613474598154,
#                                       u'http://dbpedia.org/resource/Barack_Obama_presidential_campaign'),
#                                      (0.00011295441800029948,
#                                       u'http://dbpedia.org/resource/Political_positions_of_Barack_Obama'),
#                                      (6.297003710642457e-05,
#                                       u'http://dbpedia.org/resource/Timeline_of_the_presidency_of_Barack_Obama'),
#                                      (5.6966171541716903e-05,
#                                       u'http://dbpedia.org/resource/Barack_Obama_Leadership_Academy'), (
#                                          5.1916656957473606e-05,
#                                          u'http://dbpedia.org/resource/Presidential_transition_of_Barack_Obama'),
#                                      (4.98787485412322e-05,
#                                       u'http://dbpedia.org/resource/Efforts_to_impeach_Barack_Obama'),
#                                      (4.9208945711143315e-05,
#                                       u'http://dbpedia.org/resource/Barack_Obama_Presidential_Center'),
#                                      (4.746868580696173e-05,
#                                       u'http://dbpedia.org/resource/Electoral_history_of_Barack_Obama'),
#                                      (4.603238994604908e-05,
#                                       u'http://dbpedia.org/resource/Campaign_rhetoric_of_Barack_Obama'),
#                                      (4.603238994604908e-05,
#                                       u'http://dbpedia.org/resource/Second_inauguration_of_Barack_Obama'),
#                                      (3.637570989667438e-05,
#                                       u'http://dbpedia.org/resource/Foreign_policy_of_the_Barack_Obama'),
#                                      (3.637570989667438e-05,
#                                       u'http://dbpedia.org/resource/Social_policy_of_Barack_Obama'),
#                                      (3.027915590791963e-05,
#                                       u'http://dbpedia.org/resource/First_inauguration_of_Barack_Obama')]},
#                'correct-list': {0: False, 1: False},
#                'chunktext': [
#                    {'chunk': 'parent organisation', 'surfacelength': 0, 'class': 'relation', 'surfacestart': 11},
#                    {'chunk': 'Barack Obama', 'surfacelength': 0, 'class': 'entity', 'surfacestart': 34}],
#                'types': ['relation', 'entity'], 'rejudge': False}
#         ,'Who is the parent organisation of Barack Obama?')
