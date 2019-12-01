#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
  @ Time     : 2019/4/10 19:39
  @ Author   : Vodka
  @ File     : Transform2vec.py
  @ Software : PyCharm
"""
import numpy as np
from gensim.models import KeyedVectors
from practnlptools.tools import Annotator
from predict_phrase import Predictor

class Transform2vec:
    def __init__(self):
        self.word2vec_model_path = './download/Glove/glove.6B.50d.txt'
        self.word2vec_model = KeyedVectors.load_word2vec_format(self.word2vec_model_path, binary=False,
                                                                unicode_errors='ignore')
        self.annotator = Annotator()
        self.stop_words = ["a", "as", "able", "about", "above", "according", "accordingly", "across", "actually",
                           "after", "afterwards", "again", "against", "aint", "all", "allow", "allows", "almost",
                           "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an",
                           "another", "any", "anyway", "anyways",
                           "apart", "appear", "appreciate", "appropriate", "are", "arent", "around", "as",
                           "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "be", "became",
                           "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind",
                           "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond",
                           "both", "brief", "but", "by", "cmon", "cs", "came", "can", "cant", "cannot", "cant", "cause",
                           "causes", "certain", "certainly", "changes", "clearly", "co", "com", "come", "comes",
                           "concerning", "consequently", "consider", "considering", "contain", "containing", "contains",
                           "corresponding", "could", "couldnt", "course", "currently", "definitely", "described",
                           "despite", "did", "didnt", "different", "do", "does", "doesnt", "doing", "dont", "done",
                           "down", "downwards", "during", "each", "edu", "eg", "eight", "either", "else", "elsewhere",
                           "enough", "entirely", "especially", "et", "etc", "even", "ever",
                           "ex", "exactly", "example", "except", "far", "few",
                           "ff", "fifth", "first", "five", "followed", "following", "follows", "for", "former",
                           "formerly", "forth", "four", "from", "further", "furthermore", "gets", "getting",
                           "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "had",
                           "hadnt", "hardly", "has", "hasnt", "havent", "having", "he", "hes",
                           "hello", "hence", "her", "here", "heres", "hereafter", "hereby", "herein",
                           "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully",
                           "howbeit", "however", "i", "id", "ill", "im", "ive", "ie", "if", "ignored", "immediate",
                           "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar",
                           "instead", "into", "inward", "is", "isnt", "it", "itd", "itll", "its", "its", "itself",
                           "just",  "last", "lately", "later",
                           "latter", "latterly", "least", "less", "lest", "let", "lets", "liked",
                           "likely", "little", "look", "looking", "looks", "ltd", "mainly", "many", "may", "maybe",
                           "me", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "much",
                           "must", "my", "myself", "name", "namely", "nd", "near", "nearly", "necessary",
                           "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "non",
                           "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "obviously",
                           "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto",
                           "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside",
                           "over", "overall", "own", "particular", "particularly", "per", "perhaps", "placed", "please",
                           "plus", "possible", "presumably", "probably", "provides", "que", "quite", "qv", "rather",
                           "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively",
                           "respectively", "said", "same", "saw", "saying", "says", "second",
                           "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves",
                           "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should",
                           "shouldnt", "since", "six", "so", "some",
                           "soon", "sorry", "specified", "specify",
                           "specifying", "still", "sub", "such", "sup", "sure", "ts", "take", "taken", "tell", "tends",
                           "th", "than", "thank", "thanks", "thanx", "that", "thats", "thats", "the", "their", "theirs",
                           "them", "themselves", "then", "thence", "there", "theres", "thereafter", "thereby",
                           "therefore", "therein", "theres", "thereupon", "these", "they", "theyd", "theyll", "theyre",
                           "theyve", "think", "third", "this", "thorough", "thoroughly", "those", "though", "three",
                           "through", "throughout", "thru", "thus", "to", "together", "too", "took", "toward",
                           "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "un", "under",
                           "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "used",
                           "useful", "uses", "using", "usually", "various", "very", "via", "viz", "vs", "want",
                           "wants", "was", "wasnt", "way", "we", "wed", "well", "were", "weve", "welcome", "well",
                           "went", "were", "werent",
                           "will", "willing", "wish", "with", "within", "without", "wont", "wonder", "would", "would",
                           "wouldnt", "yes", "yet", "you", "youd", "youll", "youre", "youve", "your", "yours",
                           "yourself", "yourselves", "zero", "is", ", ", "\\\\", "?", "\\"]
        self.erpredictor = Predictor()

    def MinMaxNnorm(self, arr):
        """
        Min-Max-Normalize the array
        :param arr:
        :return:normalized array
        """
        res = np.zeros(50, dtype='float32')
        for i, x in enumerate(arr):
            res[i] = float(x - np.min(arr)) / (np.max(arr) - np.min(arr))
        return res

    def transform2onevec(self, sequence):
        """
        Transform one word sequence to one vector
        :param sequence:
        :return:one vector
        """
        words = self.annotator.getAnnotations(sequence)['chunk']
        words_vec = []
        final_res = np.zeros(50, dtype='float32')
        for word in words:
            try:
                words_vec.append(self.word2vec_model.get_vector(word[0].lower()))
                final_res += self.word2vec_model.get_vector(word[0].lower())
            except:
                zeros = np.random.uniform(-1, 1, 50)
                words_vec.append(zeros)
                final_res += zeros
        final_res = self.MinMaxNnorm(final_res)
        return final_res

    def transform2vecs(self, words):
        """
        Transform word sequence(array) to some vectors(one for each)
        :param sequence:
        :return:vectors[]
        """
        words_vec = []
        for word in words:
            try:
                words_vec.append(self.word2vec_model.get_vector(word.lower()))
            except:
                print "Find new word.."
                zeros = np.random.uniform(-1, 1, 50)
                zeros.astype('float32')
                words_vec.append(zeros)
        return words_vec

    def shallowParse(self, text):
        """
        :param text: Natural Language Question
        :return: Key chunks with position infomation
        """
        result = self.annotator.getAnnotations(text)['chunk']
        chunkswithpositions = []  # For calculating surface indices
        searchfrom = 0
        for chunkpair in result:
            if chunkpair[0]!='the' and chunkpair[0]!='do' and chunkpair[0]!='does' and chunkpair[0]!='did':
                position = text.find(chunkpair[0], searchfrom)
                searchfrom = position + 1
                length = len(chunkpair[0])
                chunkswithpositions.append((chunkpair[0], chunkpair[1], position, length))
        phrases = []
        _phrase_n = []
        _phrase_v = []
        for chunk in chunkswithpositions:
            if chunk[1] == 'S-NP':
                phrases.append([chunk])
                continue
            if chunk[1] == 'B-NP' or chunk[1] == 'I-NP':
                _phrase_n.append(chunk)
                continue
            if chunk[1] == 'E-NP':
                _phrase_n.append(chunk)
                phrases.append(_phrase_n)
                _phrase_n = []
                continue
            if chunk[1] == 'S-VP':
                phrases.append([chunk])
                continue
            if chunk[1] == 'B-VP' or chunk[1] == 'I-VP':
                _phrase_v.append(chunk)
                continue
            if chunk[1] == 'E-VP':
                _phrase_v.append(chunk)
                phrases.append(_phrase_v)
                _phrase_v = []
                continue
            phrases.append([chunk])

        return phrases

    def transContext2Vecs(self, text, _token):
        """
        Return the context vectors of the _token word
        :param text:
        :param _token:
        :return:
        """
        # Filter the entities words
        result_key_chunks = self.shallowParse(text)
        combined_chunks = []
        for _index ,chunk in enumerate(result_key_chunks):
            if _index == 0:
                combined_chunks.append(chunk[0][0])
                continue
            combined_phase = []
            for word in chunk:
                if word[0].lower() != 'the':
                    combined_phase.append(word[0])
            combined_phase = ' '.join(combined_phase)
            if (combined_phase.find(_token) == -1 and _token.find(combined_phase) == -1) and (
                    self.erpredictor.predict_phrase(combined_phase) == 'R' or (chunk[0][1] != 'S-NP' and chunk[0][1] != 'B-NP' and chunk[0][1] != 'I-NP' and chunk[0][1] != 'E-NP')):
                combined_chunks.append(combined_phase)
        # print combined_chunks
        phrase = ' '.join(combined_chunks)
        words = phrase.split(' ')
        # Trans to vec
        res_vecs = self.transform2vecs(words)
        return res_vecs