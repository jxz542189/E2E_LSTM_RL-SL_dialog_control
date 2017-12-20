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
from action_manipulator import ActionManipulator

class DataUtil:
    def __init__(self):
        self.config = Config()
        # self.stops = set(stopwords.words("chinese"))
        self.lmtzr = WordNetLemmatizer()
        self.dim = self.config.embedding_size
        self.w2v_model = self.build_w2v_model()
        self.am = ActionManipulator()
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

    def get_actions(self, dialogs=[]):
        dialogs = dialogs if len(dialogs) else self.read_dialogs()
        return [row[2] for row in dialogs]

    def get_slots(self, dialogs=[]):
        dialogs = dialogs if len(dialogs) else self.read_dialogs()
        return [row[3] for row in dialogs]

    def txt_to_training_set(self):
        dialogs, dialog_indices = self.read_dialogs(with_indices=True)
        utterances = self.get_utterances(dialogs)
        responses = self.get_responses(dialogs)
        actions = self.get_actions(dialogs)
        actions = [0] + actions[:-1]
        slots = self.get_slots(dialogs)
        action_labels = [self.get_action_label(action) for action in actions]
        slot_labels = [self.get_goal_slot_label(slot) for slot in slots]
        train_set = []
        for u, r, a, s in zip(utterances, responses, action_labels, slot_labels):
            train_set.append((u, r, a, s))

        return train_set, dialog_indices

    def get_action_label(self, action_label):
        onehot = [0] * len(self.config.actions)
        onehot[action_label] = 1
        return np.array(onehot)

    def get_goal_slot_label(self, slot_labels):
        labels = np.zeros_like(self.config.slot_types)
        for i in range(len(slot_labels)):
            ind = slot_labels[i]
            slot_type_n = self.config.slot_types[ind]
            if ind >= 0:
                labels[i][ind] = 1
            else:
                for k in range(slot_type_n):
                    labels[ind][k] = 1.0/slot_type_n
        return labels


if __name__ == '__main__':
    du = DataUtil()