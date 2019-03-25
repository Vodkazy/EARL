#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
  @ Time     : 19-3-24 上午9:11
  @ Author   : Vodka
  @ File     : JointLinker.py
  @ Software : PyCharm
"""
import itertools
from pybloom_live import BloomFilter
import sys


class JointLinker:
    def __init__(self):
        print "Joint Linker initializing"
        try:
            f = open('../EARL/data/blooms/bloom1hoppredicate.pickle')
            self.bloom_1_hop_predicate = BloomFilter.fromfile(f)
            f.close()
            f = open('../EARL/data/blooms/bloom1hopentity.pickle')
            self.bloom_1_hop_entity = BloomFilter.fromfile(f)
            f.close()
            f = open('../EARL/data/blooms/bloom2hoppredicate.pickle')
            self.bloom_2_hop_predicate = BloomFilter.fromfile(f)
            f.close()
            f = open('../EARL/data/blooms/bloom2hoptypeofentity.pickle')
            self.bloom_2_hop_typeof_entity = BloomFilter.fromfile(f)
            f.close()
        except Exception, e:
            print e
            sys.exit(1)
        print "Joint Linker initialized"

    def jointLink(self, matched_chunks):
        """
        :param matched_chunks: 
        :return: 
        """
        topk_matched_list = []
        chunks = []
        types = []
        for chunk in matched_chunks:
            topk_matched_list.append(chunk['topkmatch'])
            chunks.append(chunk['chunk'])
            types.append(chunk['class'])
        nodes = {}
        cnt = 0
        for item in topk_matched_list:
            rank = 0
            nodes[cnt] = {}
            for uri in item:
                rank += 1
                if uri not in nodes[cnt]:
                    nodes[cnt][uri] = {'connections': 0, 'total_hops': 0, 'rank': rank}
            if (len(nodes[cnt]) == 0):
                nodes[cnt]['null'] = {'connections': 0, 'total_hops': 0, 'rank': 100}
            cnt += 1
        _subscript = range(len(topk_matched_list))  # Discrete the index of uris set as subscript
        # Simply loops over all possible uri pairs
        for index_pair in itertools.permutations(_subscript, 2):
            uris_1_index = index_pair[0]
            uris_2_index = index_pair[1]
            for uri_1 in topk_matched_list[uris_1_index]:
                for uri_2 in topk_matched_list[uris_2_index]:
                    blooms = uri_1 + ':' + uri_2
                    if blooms in self.bloom_1_hop_entity:
                        nodes[uris_1_index][uri_1]['connections'] += 1
                        nodes[uris_1_index][uri_1]['total_hops'] += 1
                        nodes[uris_2_index][uri_2]['connections'] += 1
                        nodes[uris_2_index][uri_2]['total_hops'] += 1
                    elif blooms in self.bloom_1_hop_predicate:
                        nodes[uris_1_index][uri_1]['connections'] += 1
                        nodes[uris_1_index][uri_1]['total_hops'] += 0.5
                        nodes[uris_2_index][uri_2]['connections'] += 1
                        nodes[uris_2_index][uri_2]['total_hops'] += 0.5
                    elif blooms in self.bloom_2_hop_typeof_entity:
                        nodes[uris_1_index][uri_1]['connections'] += 1
                        nodes[uris_1_index][uri_1]['total_hops'] += 2
                        nodes[uris_2_index][uri_2]['connections'] += 1
                        nodes[uris_2_index][uri_2]['total_hops'] += 2
                    elif blooms in self.bloom_2_hop_predicate:
                        nodes[uris_1_index][uri_1]['connections'] += 1
                        nodes[uris_1_index][uri_1]['total_hops'] += 1.5
                        nodes[uris_2_index][uri_2]['connections'] += 1
                        nodes[uris_2_index][uri_2]['total_hops'] += 1.5
        for uris_index, uris in nodes.iteritems():
            for uri, property in uris.iteritems():
                nodes[uris_index][uri]['connections'] /= float(len(topk_matched_list))
                nodes[uris_index][uri]['total_hops'] /= float(len(topk_matched_list))

        topk_res = {'nodefeatures': nodes, 'chunktext': chunks, 'types': types}

        return topk_res


if __name__ == '__main__':
    j = JointLinker()
    print j.jointLink([{'chunk': {'chunk': 'parent organisation', 'surfacelength': 19, 'class': 'relation',
                                  'surfacestart': 11}, 'topkmatch': [u'http://dbpedia.org/ontology/parentOrganisation',
                                                                     u'http://dbpedia.org/ontology/parentOrganisation',
                                                                     u'http://dbpedia.org/ontology/parentOrganisation',
                                                                     u'http://dbpedia.org/ontology/parentOrganisation',
                                                                     u'http://dbpedia.org/ontology/parentOrganisation',
                                                                     u'http://dbpedia.org/ontology/parentOrganisation',
                                                                     u'http://dbpedia.org/ontology/childOrganisation',
                                                                     u'http://dbpedia.org/ontology/childOrganisation',
                                                                     u'http://dbpedia.org/ontology/childOrganisation',
                                                                     u'http://dbpedia.org/ontology/childOrganisation',
                                                                     u'http://dbpedia.org/ontology/childOrganisation',
                                                                     u'http://dbpedia.org/ontology/childOrganisation',
                                                                     u'http://dbpedia.org/ontology/parentCompany',
                                                                     u'http://dbpedia.org/ontology/parentCompany',
                                                                     u'http://dbpedia.org/ontology/parentCompany',
                                                                     u'http://dbpedia.org/ontology/parentCompany',
                                                                     u'http://dbpedia.org/ontology/parentCompany',
                                                                     u'http://dbpedia.org/ontology/parentCompany',
                                                                     u'http://dbpedia.org/ontology/parent',
                                                                     u'http://dbpedia.org/ontology/parent',
                                                                     u'http://dbpedia.org/ontology/parent',
                                                                     u'http://dbpedia.org/ontology/parent',
                                                                     u'http://dbpedia.org/ontology/parent',
                                                                     u'http://dbpedia.org/ontology/parent',
                                                                     u'http://dbpedia.org/ontology/parent',
                                                                     u'http://dbpedia.org/ontology/parent',
                                                                     u'http://dbpedia.org/ontology/parent',
                                                                     u'http://dbpedia.org/ontology/parent',
                                                                     u'http://dbpedia.org/ontology/parent',
                                                                     u'http://dbpedia.org/ontology/parent'],
                        'class': 'relation'},
                       {'chunk': {'chunk': 'Barack Obama', 'surfacelength': 12, 'class': 'entity', 'surfacestart': 34},
                        'topkmatch': [u'http://dbpedia.org/resource/Barack_Obama_%22Hope%22_poster',
                                      u'http://dbpedia.org/resource/Barack_Obama_%22Joker%22_poster',
                                      u'http://dbpedia.org/resource/Barack_Obama',
                                      u'http://dbpedia.org/resource/Presidency_of_Barack_Obama',
                                      u'http://dbpedia.org/resource/Family_of_Barack_Obama',
                                      u'http://dbpedia.org/resource/Speeches_of_Barack_Obama',
                                      u'http://dbpedia.org/resource/Barack_Obama,_Sr.',
                                      u'http://dbpedia.org/resource/Barack_Obama_Academy',
                                      u'http://dbpedia.org/resource/Inauguration_of_Barack_Obama',
                                      u'http://dbpedia.org/resource/Barack_Obama_in_comics',
                                      u'http://dbpedia.org/resource/Barack_Obama:_The_Story',
                                      u'http://dbpedia.org/resource/The_Case_Against_Barack_Obama',
                                      u'http://dbpedia.org/resource/Barack_Obama_on_social_media',
                                      u'http://dbpedia.org/resource/Public_image_of_Barack_Obama',
                                      u"http://dbpedia.org/resource/Confirmations_of_Barack_Obama's_Cabinet",
                                      u'http://dbpedia.org/resource/Electoral_history_of_Barack_Obama',
                                      u'http://dbpedia.org/resource/Foreign_policy_of_the_Barack_Obama',
                                      u'http://dbpedia.org/resource/Social_policy_of_Barack_Obama',
                                      u'http://dbpedia.org/resource/First_inauguration_of_Barack_Obama',
                                      u'http://dbpedia.org/resource/Presidential_transition_of_Barack_Obama',
                                      u'http://dbpedia.org/resource/Barack_Obama_presidential_campaign',
                                      u"http://dbpedia.org/resource/The_Speech:_Race_and_Barack_Obama's_%22A_More_Perfect_Union%22",
                                      u'http://dbpedia.org/resource/Political_positions_of_Barack_Obama',
                                      u'http://dbpedia.org/resource/By_the_People:_The_Election_of_Barack_Obama',
                                      u'http://dbpedia.org/resource/Campaign_rhetoric_of_Barack_Obama',
                                      u'http://dbpedia.org/resource/Second_inauguration_of_Barack_Obama',
                                      u'http://dbpedia.org/resource/Barack_Obama_Presidential_Center',
                                      u'http://dbpedia.org/resource/Barack_Obama_Leadership_Academy',
                                      u'http://dbpedia.org/resource/Timeline_of_the_presidency_of_Barack_Obama',
                                      u'http://dbpedia.org/resource/Efforts_to_impeach_Barack_Obama'],
                        'class': 'entity'}]
                      )

    # Result is {
    #     'chunktext': [{'chunk': 'parent organisation', 'surfacelength': 0, 'class': 'relation', 'surfacestart': 11},
    #                   {'chunk': 'Barack Obama', 'surfacelength': 0, 'class': 'entity', 'surfacestart': 34}],
    #     'nodefeatures': {
    #         0: {u'http://dbpedia.org/ontology/parentCompany': {'connections': 0.0, 'total_hops': 0.0, 'rank': 13},
    #             u'http://dbpedia.org/ontology/parentOrganisation': {'connections': 0.0, 'total_hops': 0.0, 'rank': 1},
    #             u'http://dbpedia.org/ontology/childOrganisation': {'connections': 0.0, 'total_hops': 0.0, 'rank': 7},
    #             u'http://dbpedia.org/ontology/parent': {'connections': 12.0, 'total_hops': 6.0, 'rank': 19}},
    #         1: {
    #             u'http://dbpedia.org/resource/Barack_Obama:_The_Story': {'connections': 0.0, 'total_hops': 0.0,
    #                                                                      'rank': 11},
    #             u'http://dbpedia.org/resource/Barack_Obama,_Sr.': {'connections': 6.0, 'total_hops': 3.0, 'rank': 7},
    #             u'http://dbpedia.org/resource/Campaign_rhetoric_of_Barack_Obama': {'connections': 0.0,
    #                                                                                'total_hops': 0.0, 'rank': 25},
    #             u'http://dbpedia.org/resource/Family_of_Barack_Obama': {'connections': 6.0, 'total_hops': 3.0,
    #                                                                     'rank': 5},
    #             u'http://dbpedia.org/resource/Barack_Obama_%22Hope%22_poster': {'connections': 0.0, 'total_hops': 0.0,
    #                                                                             'rank': 1},
    #             u'http://dbpedia.org/resource/The_Case_Against_Barack_Obama': {'connections': 0.0, 'total_hops': 0.0,
    #                                                                            'rank': 12},
    #             u'http://dbpedia.org/resource/Barack_Obama_%22Joker%22_poster': {'connections': 0.0, 'total_hops': 0.0,
    #                                                                              'rank': 2},
    #             u'http://dbpedia.org/resource/Foreign_policy_of_the_Barack_Obama': {'connections': 0.0,
    #                                                                                 'total_hops': 0.0, 'rank': 17},
    #             u'http://dbpedia.org/resource/Barack_Obama_in_comics': {'connections': 0.0, 'total_hops': 0.0,
    #                                                                     'rank': 10},
    #             u'http://dbpedia.org/resource/Barack_Obama_Presidential_Center': {'connections': 0.0, 'total_hops': 0.0,
    #                                                                               'rank': 27},
    #             u'http://dbpedia.org/resource/Barack_Obama': {'connections': 0.0, 'total_hops': 0.0, 'rank': 3},
    #             u'http://dbpedia.org/resource/Efforts_to_impeach_Barack_Obama': {'connections': 0.0, 'total_hops': 0.0,
    #                                                                              'rank': 30},
    #             u'http://dbpedia.org/resource/Electoral_history_of_Barack_Obama': {'connections': 0.0,
    #                                                                                'total_hops': 0.0, 'rank': 16},
    #             u'http://dbpedia.org/resource/Presidency_of_Barack_Obama': {'connections': 0.0, 'total_hops': 0.0,
    #                                                                         'rank': 4},
    #             u'http://dbpedia.org/resource/Political_positions_of_Barack_Obama': {'connections': 0.0,
    #                                                                                  'total_hops': 0.0, 'rank': 23},
    #             u"http://dbpedia.org/resource/The_Speech:_Race_and_Barack_Obama's_%22A_More_Perfect_Union%22": {
    #                 'connections': 0.0, 'total_hops': 0.0, 'rank': 22},
    #             u"http://dbpedia.org/resource/Confirmations_of_Barack_Obama's_Cabinet": {'connections': 0.0,
    #                                                                                      'total_hops': 0.0, 'rank': 15},
    #             u'http://dbpedia.org/resource/Public_image_of_Barack_Obama': {'connections': 0.0, 'total_hops': 0.0,
    #                                                                           'rank': 14},
    #             u'http://dbpedia.org/resource/Barack_Obama_on_social_media': {'connections': 0.0, 'total_hops': 0.0,
    #                                                                           'rank': 13},
    #             u'http://dbpedia.org/resource/Social_policy_of_Barack_Obama': {'connections': 0.0, 'total_hops': 0.0,
    #                                                                            'rank': 18},
    #             u'http://dbpedia.org/resource/Second_inauguration_of_Barack_Obama': {'connections': 0.0,
    #                                                                                  'total_hops': 0.0, 'rank': 26},
    #             u'http://dbpedia.org/resource/By_the_People:_The_Election_of_Barack_Obama': {'connections': 0.0,
    #                                                                                          'total_hops': 0.0,
    #                                                                                          'rank': 24},
    #             u'http://dbpedia.org/resource/Inauguration_of_Barack_Obama': {'connections': 0.0, 'total_hops': 0.0,
    #                                                                           'rank': 9},
    #             u'http://dbpedia.org/resource/Barack_Obama_Academy': {'connections': 0.0, 'total_hops': 0.0, 'rank': 8},
    #             u'http://dbpedia.org/resource/Barack_Obama_Leadership_Academy': {'connections': 0.0, 'total_hops': 0.0,
    #                                                                              'rank': 28},
    #             u'http://dbpedia.org/resource/Presidential_transition_of_Barack_Obama': {'connections': 0.0,
    #                                                                                      'total_hops': 0.0, 'rank': 20},
    #             u'http://dbpedia.org/resource/Barack_Obama_presidential_campaign': {'connections': 0.0,
    #                                                                                 'total_hops': 0.0, 'rank': 21},
    #             u'http://dbpedia.org/resource/Speeches_of_Barack_Obama': {'connections': 0.0, 'total_hops': 0.0,
    #                                                                       'rank': 6},
    #             u'http://dbpedia.org/resource/First_inauguration_of_Barack_Obama': {'connections': 0.0,
    #                                                                                 'total_hops': 0.0, 'rank': 19},
    #             u'http://dbpedia.org/resource/Timeline_of_the_presidency_of_Barack_Obama': {'connections': 0.0,
    #                                                                                         'total_hops': 0.0,
    #                                                                                         'rank': 29}}},
    #     'types': ['relation', 'entity']}
