#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
  @ Time     : 19-3-25 下午2:15
  @ Author   : Vodka
  @ File     : ReRanker.py
  @ Software : PyCharm
"""
import xgboost as xgb
import numpy as np
import sys
import editdistance


class ReRanker:
    def __init__(self):
        print "ReRanker initializing"
        try:
            self.change_type_arr = {}
            self.change_flag = False
            self.model = xgb.Booster({'nthread': 4})
            self.model.load_model('model/reranker.model')
        except Exception, e:
            print e
            sys.exit(1)
        print "ReRanker initialized"

    def reRank(self, topk_res):
        rerankedlists = {}
        for _index, uris_with_features in topk_res['nodefeatures'].iteritems():  # travel all the nodes{0:'xxxxx'}
            features = []
            uris = []
            distance = []
            for uri, feature in uris_with_features.iteritems():
                uris.append(uri)
                ontology_name = uri.split('/')[-1].lower()
                question_name = topk_res['chunktext'][_index]['chunk'].lower()
                features.append([feature['connections'], feature['total_hops'], feature['rank']])  # change {} into []
                distance.append(editdistance.eval(ontology_name, question_name))
                # print ontology_name,question_name,editdistance.eval(ontology_name,question_name)
            features = np.array(features)
            _input = xgb.DMatrix(features)
            _output = self.model.predict(_input)
            prediction = (np.max(_output))
            # print _output
            if prediction < 0.1 and topk_res['types'][_index] == 'relation' and np.min(distance) > 1:
                print "Change type of " + topk_res['chunktext'][_index]['chunk'] + " to entity"
                self.change_type_arr[_index] = True
                self.change_flag = True
            else:
                self.change_type_arr[_index] = False
            score_uri_list = [(float(score), uri) for score, uri in zip(_output, uris)]
            rerankedlists[_index] = sorted(score_uri_list, key=lambda x: x[0], reverse=True)
        return {'rerankedlists': rerankedlists, 'chunktext': topk_res['chunktext'], 'types': topk_res['types'],
                'rejudge': self.change_flag, 'correct-list': self.change_type_arr}


if __name__ == '__main__':
    r = ReRanker()
    print r.reRank({
        'chunktext': [{'chunk': 'parent organisation', 'surfacelength': 0, 'class': 'relation', 'surfacestart': 11},
                      {'chunk': 'Barack Obama', 'surfacelength': 0, 'class': 'entity', 'surfacestart': 34}],
        'nodefeatures': {
            0: {u'http://dbpedia.org/ontology/parentCompany': {'connections': 0.0, 'total_hops': 0.0, 'rank': 13},
                u'http://dbpedia.org/ontology/parentOrganisation': {'connections': 0.0, 'total_hops': 0.0, 'rank': 1},
                u'http://dbpedia.org/ontology/childOrganisation': {'connections': 0.0, 'total_hops': 0.0, 'rank': 7},
                u'http://dbpedia.org/ontology/parent': {'connections': 12.0, 'total_hops': 6.0, 'rank': 19}}, 1: {
                u'http://dbpedia.org/resource/Barack_Obama:_The_Story': {'connections': 0.0, 'total_hops': 0.0,
                                                                         'rank': 11},
                u'http://dbpedia.org/resource/Barack_Obama,_Sr.': {'connections': 6.0, 'total_hops': 3.0, 'rank': 7},
                u'http://dbpedia.org/resource/Campaign_rhetoric_of_Barack_Obama': {'connections': 0.0,
                                                                                   'total_hops': 0.0, 'rank': 25},
                u'http://dbpedia.org/resource/Family_of_Barack_Obama': {'connections': 6.0, 'total_hops': 3.0,
                                                                        'rank': 5},
                u'http://dbpedia.org/resource/Barack_Obama_%22Hope%22_poster': {'connections': 0.0, 'total_hops': 0.0,
                                                                                'rank': 1},
                u'http://dbpedia.org/resource/The_Case_Against_Barack_Obama': {'connections': 0.0, 'total_hops': 0.0,
                                                                               'rank': 12},
                u'http://dbpedia.org/resource/Barack_Obama_%22Joker%22_poster': {'connections': 0.0, 'total_hops': 0.0,
                                                                                 'rank': 2},
                u'http://dbpedia.org/resource/Foreign_policy_of_the_Barack_Obama': {'connections': 0.0,
                                                                                    'total_hops': 0.0, 'rank': 17},
                u'http://dbpedia.org/resource/Barack_Obama_in_comics': {'connections': 0.0, 'total_hops': 0.0,
                                                                        'rank': 10},
                u'http://dbpedia.org/resource/Barack_Obama_Presidential_Center': {'connections': 0.0, 'total_hops': 0.0,
                                                                                  'rank': 27},
                u'http://dbpedia.org/resource/Barack_Obama': {'connections': 0.0, 'total_hops': 0.0, 'rank': 3},
                u'http://dbpedia.org/resource/Efforts_to_impeach_Barack_Obama': {'connections': 0.0, 'total_hops': 0.0,
                                                                                 'rank': 30},
                u'http://dbpedia.org/resource/Electoral_history_of_Barack_Obama': {'connections': 0.0,
                                                                                   'total_hops': 0.0, 'rank': 16},
                u'http://dbpedia.org/resource/Presidency_of_Barack_Obama': {'connections': 0.0, 'total_hops': 0.0,
                                                                            'rank': 4},
                u'http://dbpedia.org/resource/Political_positions_of_Barack_Obama': {'connections': 0.0,
                                                                                     'total_hops': 0.0, 'rank': 23},
                u"http://dbpedia.org/resource/The_Speech:_Race_and_Barack_Obama's_%22A_More_Perfect_Union%22": {
                    'connections': 0.0, 'total_hops': 0.0, 'rank': 22},
                u"http://dbpedia.org/resource/Confirmations_of_Barack_Obama's_Cabinet": {'connections': 0.0,
                                                                                         'total_hops': 0.0, 'rank': 15},
                u'http://dbpedia.org/resource/Public_image_of_Barack_Obama': {'connections': 0.0, 'total_hops': 0.0,
                                                                              'rank': 14},
                u'http://dbpedia.org/resource/Barack_Obama_on_social_media': {'connections': 0.0, 'total_hops': 0.0,
                                                                              'rank': 13},
                u'http://dbpedia.org/resource/Social_policy_of_Barack_Obama': {'connections': 0.0, 'total_hops': 0.0,
                                                                               'rank': 18},
                u'http://dbpedia.org/resource/Second_inauguration_of_Barack_Obama': {'connections': 0.0,
                                                                                     'total_hops': 0.0, 'rank': 26},
                u'http://dbpedia.org/resource/By_the_People:_The_Election_of_Barack_Obama': {'connections': 0.0,
                                                                                             'total_hops': 0.0,
                                                                                             'rank': 24},
                u'http://dbpedia.org/resource/Inauguration_of_Barack_Obama': {'connections': 0.0, 'total_hops': 0.0,
                                                                              'rank': 9},
                u'http://dbpedia.org/resource/Barack_Obama_Academy': {'connections': 0.0, 'total_hops': 0.0, 'rank': 8},
                u'http://dbpedia.org/resource/Barack_Obama_Leadership_Academy': {'connections': 0.0, 'total_hops': 0.0,
                                                                                 'rank': 28},
                u'http://dbpedia.org/resource/Presidential_transition_of_Barack_Obama': {'connections': 0.0,
                                                                                         'total_hops': 0.0, 'rank': 20},
                u'http://dbpedia.org/resource/Barack_Obama_presidential_campaign': {'connections': 0.0,
                                                                                    'total_hops': 0.0, 'rank': 21},
                u'http://dbpedia.org/resource/Speeches_of_Barack_Obama': {'connections': 0.0, 'total_hops': 0.0,
                                                                          'rank': 6},
                u'http://dbpedia.org/resource/First_inauguration_of_Barack_Obama': {'connections': 0.0,
                                                                                    'total_hops': 0.0, 'rank': 19},
                u'http://dbpedia.org/resource/Timeline_of_the_presidency_of_Barack_Obama': {'connections': 0.0,
                                                                                            'total_hops': 0.0,
                                                                                            'rank': 29}}},
        'types': ['relation', 'entity']})

    # Result is {'rerankedlists': {1: [(0.25071218609809875, u'http://dbpedia.org/resource/Family_of_Barack_Obama'),
    #                                  (0.1938347965478897, u'http://dbpedia.org/resource/Barack_Obama,_Sr.'),
    #                                  (0.007336166687309742,
    #                                   u'http://dbpedia.org/resource/Barack_Obama_%22Hope%22_poster'),
    #                                  (0.004728458821773529,
    #                                   u'http://dbpedia.org/resource/Barack_Obama_%22Joker%22_poster'),
    #                                  (0.0014046295545995235, u'http://dbpedia.org/resource/Barack_Obama'),
    #                                  (0.0011621536687016487, u'http://dbpedia.org/resource/Presidency_of_Barack_Obama'),
    #                                  (0.0004659400146920234,
    #                                   u'http://dbpedia.org/resource/The_Case_Against_Barack_Obama'), (
    #                                      0.000256964034633711,
    #                                      u"http://dbpedia.org/resource/Confirmations_of_Barack_Obama's_Cabinet"),
    #                                  (0.0002468969614710659, u'http://dbpedia.org/resource/Barack_Obama:_The_Story'),
    #                                  (0.00024368573212996125, u'http://dbpedia.org/resource/Speeches_of_Barack_Obama'),
    #                                  (0.0002059761609416455,
    #                                   u'http://dbpedia.org/resource/Barack_Obama_on_social_media'), (
    #                                      0.00017089983157347888,
    #                                      u'http://dbpedia.org/resource/By_the_People:_The_Election_of_Barack_Obama'),
    #                                  (0.00015553012781310827,
    #                                   u'http://dbpedia.org/resource/Public_image_of_Barack_Obama'),
    #                                  (0.00015441638242918998, u'http://dbpedia.org/resource/Barack_Obama_Academy'),
    #                                  (0.0001491083821747452,
    #                                   u'http://dbpedia.org/resource/Inauguration_of_Barack_Obama'),
    #                                  (0.00012262043310329318, u'http://dbpedia.org/resource/Barack_Obama_in_comics'), (
    #                                      0.00011838223872473463,
    #                                      u"http://dbpedia.org/resource/The_Speech:_Race_and_Barack_Obama's_%22A_More_Perfect_Union%22"),
    #                                  (0.00011634613474598154,
    #                                   u'http://dbpedia.org/resource/Barack_Obama_presidential_campaign'),
    #                                  (0.00011295441800029948,
    #                                   u'http://dbpedia.org/resource/Political_positions_of_Barack_Obama'),
    #                                  (6.297003710642457e-05,
    #                                   u'http://dbpedia.org/resource/Timeline_of_the_presidency_of_Barack_Obama'),
    #                                  (5.6966171541716903e-05,
    #                                   u'http://dbpedia.org/resource/Barack_Obama_Leadership_Academy'), (
    #                                      5.1916656957473606e-05,
    #                                      u'http://dbpedia.org/resource/Presidential_transition_of_Barack_Obama'),
    #                                  (4.98787485412322e-05,
    #                                   u'http://dbpedia.org/resource/Efforts_to_impeach_Barack_Obama'),
    #                                  (4.9208945711143315e-05,
    #                                   u'http://dbpedia.org/resource/Barack_Obama_Presidential_Center'),
    #                                  (4.746868580696173e-05,
    #                                   u'http://dbpedia.org/resource/Electoral_history_of_Barack_Obama'),
    #                                  (4.603238994604908e-05,
    #                                   u'http://dbpedia.org/resource/Campaign_rhetoric_of_Barack_Obama'),
    #                                  (4.603238994604908e-05,
    #                                   u'http://dbpedia.org/resource/Second_inauguration_of_Barack_Obama'),
    #                                  (3.637570989667438e-05,
    #                                   u'http://dbpedia.org/resource/Foreign_policy_of_the_Barack_Obama'),
    #                                  (3.637570989667438e-05,
    #                                   u'http://dbpedia.org/resource/Social_policy_of_Barack_Obama'),
    #                                  (3.027915590791963e-05,
    #                                   u'http://dbpedia.org/resource/First_inauguration_of_Barack_Obama')]},
    #            'correct-list': {0: False, 1: False},
    #            'chunktext': [
    #                {'chunk': 'parent organisation', 'surfacelength': 0, 'class': 'relation', 'surfacestart': 11},
    #                {'chunk': 'Barack Obama', 'surfacelength': 0, 'class': 'entity', 'surfacestart': 34}],
    #            'types': ['relation', 'entity'], 'rejudge': False}
