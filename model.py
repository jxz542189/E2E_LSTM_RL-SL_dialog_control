import keras
from data_util import DataUtil
from config import Config
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Bidirectional, LSTM, Input, Concatenate
import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model

class DialogueStateModel:
    def __init__(self):
        self.config = Config()
        self.du = DataUtil()

    def build_model(self):
        tokens = Input(shape=(self.config.max_turn, self.config.embedding_size))
        prev_action = Input(shape=(1))
        utterance = Bidirectional(LSTM(self.config.u_state_size, return_sequences=True))(tokens)
        x = Concatenate([utterance, prev_action])
        dialogue_state = Bidirectional(LSTM(self.config.d_state_size))(x)
        slot_value_logits = Dense(self.config.n_slots, activation='softmax')(dialogue_state)

        keras.layers.Dropout(self.config.dropout)

