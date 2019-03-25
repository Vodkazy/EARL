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


class TextMatch:
    def __init__(self):
        print "TextMatch initializing"
        try:
            self.es = Elasticsearch()
            self.label2uri = {}
            self.cache = {}
            file = open('../EARL/data/ontologylabeluridict.json')
            buff_r = file.read()
            self.label2uri = json.loads(buff_r)
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

    def textMatch(self, chunks):
        """
        :param chunks: 
        :return: 
        """
        matched_chunks = []
        for chunk in chunks:
            # Use the Elasticsearch to generate the candidate uri of entity
            if chunk['class'] == 'entity':
                res = self.es.search(index="dbentityindex11", doc_type="records", body={
                    "query": {
                        "multi_match": {"query": chunk['chunk'], "fields": ["wikidataLabel", "dbpediaLabel^1.5"]}},
                    "size": 500})
                temp_topk = []
                res_topk = []
                for record in res['hits']['hits']:
                    temp_topk.append((record['_source']['uri'], record['_source']['edgecount']))
                # temp_topk = sorted(temp_topk, key=lambda k: k[1], reverse=True)
                for record in temp_topk:
                    if (len(res_topk) >= 30):
                        break
                    if record[0] in res_topk:
                        continue
                    else:
                        res_topk.append(record[0])
                matched_chunks.append({'chunk': chunk, 'topkmatch': res_topk, 'class': 'entity'})

            # Use the label2uri dictionary to generate the candidate uri of relation
            if chunk['class'] == 'relation':
                phrase = chunk['chunk']
                res = []
                for _key, _value in self.label2uri.iteritems():
                    score = self.pharse_similarity(_key, phrase)
                    res.append({'label': _key, 'score': float(score), 'uri': _value})
                res = sorted(res, key=lambda v: v['score'], reverse=True)
                uris = []
                for _ in res:
                    if _['uri'] in uris:
                        continue
                    else:
                        # uris.append(_['uri']) This form is wrong
                        uris += _['uri']
                uris = uris[:30]
                matched_chunks.append({'chunk': chunk, 'topkmatch': uris, 'class': 'relation'})
        return matched_chunks


if __name__ == '__main__':
    t = TextMatch()
    print t.textMatch([{'chunk': 'parent organisation', 'surfacelength': 19, 'class': 'relation', 'surfacestart': 11},
                       {'chunk': 'Barack Obama', 'surfacelength': 12, 'class': 'entity', 'surfacestart': 34}]
                      )

    # Result is [{'chunk': {'chunk': 'parent organisation', 'surfacelength': 19, 'class': 'relation', 'surfacestart': 11},
    #             'topkmatch': [u'http://dbpedia.org/ontology/parentOrganisation',
    #                           u'http://dbpedia.org/ontology/parentOrganisation',
    #                           u'http://dbpedia.org/ontology/parentOrganisation',
    #                           u'http://dbpedia.org/ontology/parentOrganisation',
    #                           u'http://dbpedia.org/ontology/parentOrganisation',
    #                           u'http://dbpedia.org/ontology/parentOrganisation',
    #                           u'http://dbpedia.org/ontology/childOrganisation',
    #                           u'http://dbpedia.org/ontology/childOrganisation',
    #                           u'http://dbpedia.org/ontology/childOrganisation',
    #                           u'http://dbpedia.org/ontology/childOrganisation',
    #                           u'http://dbpedia.org/ontology/childOrganisation',
    #                           u'http://dbpedia.org/ontology/childOrganisation',
    #                           u'http://dbpedia.org/ontology/parentCompany',
    #                           u'http://dbpedia.org/ontology/parentCompany',
    #                           u'http://dbpedia.org/ontology/parentCompany',
    #                           u'http://dbpedia.org/ontology/parentCompany',
    #                           u'http://dbpedia.org/ontology/parentCompany',
    #                           u'http://dbpedia.org/ontology/parentCompany', u'http://dbpedia.org/ontology/parent',
    #                           u'http://dbpedia.org/ontology/parent', u'http://dbpedia.org/ontology/parent',
    #                           u'http://dbpedia.org/ontology/parent', u'http://dbpedia.org/ontology/parent',
    #                           u'http://dbpedia.org/ontology/parent', u'http://dbpedia.org/ontology/parent',
    #                           u'http://dbpedia.org/ontology/parent', u'http://dbpedia.org/ontology/parent',
    #                           u'http://dbpedia.org/ontology/parent', u'http://dbpedia.org/ontology/parent',
    #                           u'http://dbpedia.org/ontology/parent'], 'class': 'relation'},
    #            {'chunk': {'chunk': 'Barack Obama', 'surfacelength': 12, 'class': 'entity', 'surfacestart': 34},
    #             'topkmatch': [u'http://dbpedia.org/resource/Barack_Obama_%22Hope%22_poster',
    #                           u'http://dbpedia.org/resource/Barack_Obama_%22Joker%22_poster',
    #                           u'http://dbpedia.org/resource/Barack_Obama',
    #                           u'http://dbpedia.org/resource/Presidency_of_Barack_Obama',
    #                           u'http://dbpedia.org/resource/Family_of_Barack_Obama',
    #                           u'http://dbpedia.org/resource/Speeches_of_Barack_Obama',
    #                           u'http://dbpedia.org/resource/Barack_Obama,_Sr.',
    #                           u'http://dbpedia.org/resource/Barack_Obama_Academy',
    #                           u'http://dbpedia.org/resource/Inauguration_of_Barack_Obama',
    #                           u'http://dbpedia.org/resource/Barack_Obama_in_comics',
    #                           u'http://dbpedia.org/resource/Barack_Obama:_The_Story',
    #                           u'http://dbpedia.org/resource/The_Case_Against_Barack_Obama',
    #                           u'http://dbpedia.org/resource/Barack_Obama_on_social_media',
    #                           u'http://dbpedia.org/resource/Public_image_of_Barack_Obama',
    #                           u"http://dbpedia.org/resource/Confirmations_of_Barack_Obama's_Cabinet",
    #                           u'http://dbpedia.org/resource/Electoral_history_of_Barack_Obama',
    #                           u'http://dbpedia.org/resource/Foreign_policy_of_the_Barack_Obama',
    #                           u'http://dbpedia.org/resource/Social_policy_of_Barack_Obama',
    #                           u'http://dbpedia.org/resource/First_inauguration_of_Barack_Obama',
    #                           u'http://dbpedia.org/resource/Presidential_transition_of_Barack_Obama',
    #                           u'http://dbpedia.org/resource/Barack_Obama_presidential_campaign',
    #                           u"http://dbpedia.org/resource/The_Speech:_Race_and_Barack_Obama's_%22A_More_Perfect_Union%22",
    #                           u'http://dbpedia.org/resource/Political_positions_of_Barack_Obama',
    #                           u'http://dbpedia.org/resource/By_the_People:_The_Election_of_Barack_Obama',
    #                           u'http://dbpedia.org/resource/Campaign_rhetoric_of_Barack_Obama',
    #                           u'http://dbpedia.org/resource/Second_inauguration_of_Barack_Obama',
    #                           u'http://dbpedia.org/resource/Barack_Obama_Presidential_Center',
    #                           u'http://dbpedia.org/resource/Barack_Obama_Leadership_Academy',
    #                           u'http://dbpedia.org/resource/Timeline_of_the_presidency_of_Barack_Obama',
    #                           u'http://dbpedia.org/resource/Efforts_to_impeach_Barack_Obama'], 'class': 'entity'}]
