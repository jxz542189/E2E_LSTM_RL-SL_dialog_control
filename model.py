import keras
from data_util import DataUtil
from config import Config
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, Bidirectional, LSTM, Input, Concatenate, Lambda
import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.models import Model

class DialogueStateModel:
    def __init__(self, load=False):
        self.config = Config()
        self.du = DataUtil()
        self.load = load

    def build_model_google(self):
        tokens = Input(shape=(self.config.max_turn, self.config.embedding_size))
        prev_action = Input(shape=(1))
        utterance = Bidirectional(LSTM(self.config.u_state_size, return_sequences=True, dropout=self.config.dropout))(tokens)
        x = Concatenate([utterance, prev_action])
        dialogue_state = Bidirectional(LSTM(self.config.d_state_size, dropout=self.config.dropout))(x)
        slot1_hidden = Dense(self.config.h1, activation='relu')(dialogue_state)
        slot1_value_logits = Dense(self.config.slot_types[0], activation='softmax')(slot1_hidden)
        slot2_hidden = Dense(self.config.h1, activation='relu')(dialogue_state)
        slot2_value_logits = Dense(self.config.slot_types[1], activation='softmax')(slot2_hidden)
        slot3_hidden = Dense(self.config.h1, activation='relu')(dialogue_state)
        slot3_value_logits = Dense(self.config.slot_types[2], activation='softmax')(slot3_hidden)
        slot_value_logits = Concatenate([slot1_value_logits, slot2_value_logits, slot3_value_logits])
        slot1_log = Lambda(lambda x:np.log(x))(slot1_value_logits)
        slot2_log = Lambda(lambda x:np.log(x))(slot2_value_logits)
        slot3_log = Lambda(lambda x:np.log(x))(slot3_value_logits)
        log_slot_value_logits = Concatenate([slot1_log, slot2_log, slot3_log])
        policy_input = Concatenate([dialogue_state, log_slot_value_logits])
        policy_hidden = Dense(self.config.h1, activation='relu')(policy_input)
        action_logits = Dense(len(self.config.actions), activation='softmax')(policy_hidden)
        model = Model(inputs=[tokens, prev_action], outputs=[slot_value_logits, action_logits])
        return model

    def train(self):
        if not self.load:
            model = self.build_model_google()
            model.compile(loss=self.additive_cross_entropy,
                          optimizer=keras.optimizers.adadelta(lr=self.config.init_lr),
                          metrics=['accuracy'])


    def additive_cross_entropy(self, y_true, y_pred):
        true_slot_labels = y_true[0]
        pred_slot_labels = y_pred[0]
        true_action_labels = y_true[1]
        pred_action_labels = y_pred[1]
        ce1 = keras.backend.categorical_crossentropy(true_slot_labels, pred_slot_labels)
        ce2 = keras.backend.categorical_crossentropy(true_action_labels, pred_action_labels)
        return ce1 + ce2

