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
        self.tagger = StanfordNERTagger(model_filename=self.config.ner_model_path, path_to_jar=self.config.ner_jar_path)
        self.stops = set(stopwords.words("english"))
        self.lmtzr = WordNetLemmatizer()
        self.postagger = StanfordPOSTagger(path_to_jar=self.config.pos_jar_path, model_filename=self.config.pos_model_path)
        self.dependency_parser = StanfordDependencyParser(path_to_jar=self.config.dep_jar_path,
                                                          path_to_models_jar=self.config.dep_model_path)

# *********************************** FOR LOAD DATA & MODELS *********************************************


    def load_word2vec_model(self):
        self.model = KeyedVectors.load_word2vec_format(self.config.word2vec_model_path, binary=False)


    def load_json(self, fname):
        f = codecs.open(fname, 'rb', encoding='utf-8')
        print fname + ' done'
        data = json.load(f)
        qas = [doc['qa'] for doc in data]
        wikis = [doc['sentences'] for doc in data]
        questions = [[qa['question'] for qa in qa_list] for qa_list in qas]
        if 'test' in fname:
            answer_indices = []
            answers = []
        else:
            answers = [[qa['answer'] for qa in qa_list] for qa_list in qas]
            answer_indices = [[qa['answer_sentence'] for qa in qa_list] for qa_list in qas]
        return wikis, questions, answers, answer_indices


# *********************************** FOR PREPROCESSING *********************************************


    def lemmatize(self, word):
        word = word.lower()
        lemma = self.lmtzr.lemmatize(word, 'v')
        if lemma == word:
            lemma = self.lmtzr.lemmatize(word, 'n')
        return lemma

    def lemmatize_sent(self, words):
        return [self.lemmatize(word) for word in words]

    def remove_non_alphanumeric(self, words):
        return [re.sub(r'''([^\s\w])+''', '', word) for word in words]

    def ner_sent(self, words):
        ner_sent = self.tagger.tag(words)
        return ner_sent

    def lower_sent(self, words):
        return [word.lower() for word in words]

    def remove_stop_words(self, words):
        return [word for word in words if word.lower() not in self.stops]

    def remove_mark(self, word):
        return ''.join([c for c in word if c not in punctuation])

    def is_all_puncs(self, token):
        if all([x in punctuation for x in token]):
            return True
        return False

    def remove_punc_in_token(self, token):
        return ''.join([x for x in token if x not in punctuation]).strip()

    def preprocess_wiki(self, wiki):
        raw_split = [word_tokenize(sent.replace(u"\u200b",'')) for sent in wiki]
        remove_pure_punc = [[token for token in sent if not self.is_all_puncs(token)] for sent in raw_split]
        remove_punc_in_words = [[self.remove_punc_in_token(token) for token in sent] for sent in remove_pure_punc]
        ner = self.ner_tagging(remove_punc_in_words)
        lower = [self.lower_sent(sent) for sent in remove_punc_in_words]
        remove_stop = [self.remove_stop_words(sent) for sent in lower]
        lemmatized = [self.lemmatize_sent(sent) for sent in remove_stop]
        return remove_pure_punc, ner, lemmatized


if __name__ == '__main__':
    du = DataUtil()