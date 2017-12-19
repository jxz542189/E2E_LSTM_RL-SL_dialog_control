# -*- coding: utf-8 -*-
from gensim.models import Word2Vec
from numpy import ndarray as nd
import numpy as np
from config import Config
import json
import nltk
from collections import defaultdict
from nltk.corpus import stopwords
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.stanford import StanfordNERTagger
from string import punctuation
from nltk import word_tokenize
from nltk.parse.stanford import StanfordDependencyParser
from nltk.tag.stanford import StanfordPOSTagger
import math
import codecs
import pickle
from math import log
from gensim.models.keyedvectors import KeyedVectors
from string import punctuation
from nltk import pos_tag, pos_tag_sents


class DataUtil:
    def __init__(self):
        self.config = Config()
        # self.stops = set(stopwords.words("chinese"))
        self.lmtzr = WordNetLemmatizer()
        self.dim = self.config.embedding_size
        self.w2v_model = self.build_w2v_model()

# *********************************** FOR LOAD DATA & MODELS *********************************************
    def build_w2v_model(self, load=False):
        if not load:
            model = self.load_w2v_model()
            self.save_model(model)
        else:
            model = self.load_model()
        return model

    def save_model(self, model):
        path = 'data/chinese.model'
        with open(path, 'w') as f:
            pickle.dump(model, f)
        print 'model saved'

    def load_model(self, path='data/chinese.model'):
        with open(path, 'r') as f:
            model = pickle.load(f)
        return model

    def encode_seq(self, seq):
        len_diff = self.config.max_sent_len - len(seq)
        embs = [ self.w2v_model[word] for word in seq.split(' ') if word and word in self.w2v_model]
        embs.extend(len_diff*[0.0])
        return np.array(embs)



    def encode_by_averaging(self, utterance):
        embs = [ self.w2v_model[word] for word in utterance.split(' ') if word and word in self.w2v_model]
        # average of embeddings
        if len(embs):
            return np.mean(embs, axis=0)
        else:
            return np.zeros([self.dim],np.float32)

    def load_w2v_model(self, fname="data/newsblogbbs.vec", load=False):
        print "load model..."
        with codecs.open(fname, 'r', "utf-8") as f:
            vocab = self.load_w2v_data(f)
        print vocab['</s>']
        return vocab

    def load_w2v_data(self, f):
        size = 0
        # vocab = []
        # feature = []
        vocab_dict = {}
        flag = 0
        while True:
            line = f.readline()
            if not line:
                break
            if flag == 0:
                line = line.strip().split()
                _, size = int(line[0]), int(line[1])
                flag = 1
                continue
            line = line.strip().split()
            if not line:
                continue
            w = line[0]
            vec = [float(i) for i in line[1:]]
            if len(vec) != size:
                continue
            vec = np.array(vec)
            length = np.sqrt((vec ** 2).sum())
            vec /= length
            # print length,vec
            vocab_dict[w] = vec
        return vocab_dict


    def read_dialogs(self, fname='data/dialog_train.txt', with_indices=False):
        def rm_index(row):
            return [' '.join(row[0].split(' ')[1:])] + row[1:]

        def filter_(dialogs):
            filtered_ = []
            for row in dialogs:
                if row[0][:6] != 'service_':
                    filtered_.append(row)
            return filtered_

        with open(fname) as f:
            dialogs = filter_([ rm_index(row.split('\t')) for row in  f.read().split('\n') ])
            # organize dialogs -> dialog_indices
            prev_idx = -1
            n = 1
            dialog_indices = []
            updated_dialogs = []
            for i, dialog in enumerate(dialogs):
                if not dialogs[i][0]:
                    dialog_indices.append({
                        'start' : prev_idx + 1,
                        'end' : i - n + 1
                    })
                    prev_idx = i-n
                    n += 1
                else:
                    updated_dialogs.append(dialog)
            if with_indices:
                return updated_dialogs, dialog_indices[:-1]

            return updated_dialogs

    def get_utterances(self, dialogs=[]):
        dialogs = dialogs if len(dialogs) else self.read_dialogs()
        return [row[0] for row in dialogs]

    def get_responses(self, dialogs=[]):
        dialogs = dialogs if len(dialogs) else self.read_dialogs()
        return [row[1] for row in dialogs]

    def utterance_to_embed(self, ut):
        return []
    '''
        Train

        1. Prepare training examples
            1.1 Format 'utterance \t action_template_id\n'
        2. Prepare dev set
        3. Organize trainset as list of dialogues
    '''

    # class Data():
    #
    #     def __init__(self, entity_tracker, action_tracker):
    #
    #         self.action_templates = action_tracker.get_action_templates()
    #         self.et = entity_tracker
    #         # prepare data
    #         self.trainset = self.prepare_data()
    #
    #     def prepare_data(self):
    #         # get dialogs from file
    #         dialogs, dialog_indices = util.read_dialogs(with_indices=True)
    #         # get utterances
    #         utterances = util.get_utterances(dialogs)
    #         # get responses
    #         responses = util.get_responses(dialogs)
    #         responses = [self.get_template_id(response) for response in responses]
    #
    #         trainset = []
    #         for u, r in zip(utterances, responses):
    #             trainset.append((u, r))
    #
    #         return trainset, dialog_indices
    #
    #     def get_template_id(self, response):
    #
    #         def extract_(response):
    #             template = []
    #             for word in response.split(' '):
    #                 if 'resto_' in word:
    #                     if 'phone' in word:
    #                         template.append('<info_phone>')
    #                     elif 'address' in word:
    #                         template.append('<info_address>')
    #                     else:
    #                         template.append('<restaurant>')
    #                 else:
    #                     template.append(word)
    #             return ' '.join(template)
    #
    #         return self.action_templates.index(
    #             extract_(self.et.extract_entities(response, update=False))
    #         )


if __name__ == '__main__':
    du = DataUtil()