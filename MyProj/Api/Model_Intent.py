import numpy as np
import tensorflow as tf
import pandas as pd
import os
import json
import re
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate
from tensorflow.keras.layers import Layer
from transformers import TFAutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP


graph = tf.compat.v1.reset_default_graph()

class Attention(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, hidden_states):
        hidden_size = int(hidden_states.shape[2])
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = dot([score_first_part, h_t], [2, 1], name='attention_score')
        attention_weights = Activation('softmax', name='attention_weight')(score)
        context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
        pre_activation = concatenate([context_vector, h_t], name='attention_output')
        attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector

class Model_Cls:
    def __init__(self,batch_size = 64, epochs = 10):
        self.batch_size = batch_size
        self.epochs = epochs
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        self.rdrsegmenter = VnCoreNLP("MyProj/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
    def cleaning(self,sentences):
        clean = re.sub(r'[!"#$%&()*+,-./:;<=>?@[\]^`{|}~]\s*', " ", sentences)
        clean = clean.strip()
        return clean


    def pre_processing(self,text):
            Raw_X = []
            text = self.cleaning(text)
            sents = self.rdrsegmenter.tokenize(text)
            tmp = ''
            # print(text)
            for sent in sents[0]:
                tmp += sent+' '
            tmp = tmp.strip()
            Raw_X.append(tmp)
            return Raw_X
    def encoding(self,sent,max_length = 20):
        
        all_sent = []
        all_mask_sent = []
        for line in sent:
            
            # print(line)
            # l = tokenizer.encode(line)
            tokens = self.tokenizer.encode_plus(line, max_length=max_length,
                                        truncation=True, padding='max_length',
                                        add_special_tokens=True, return_attention_mask=True,
                                        return_token_type_ids=False, return_tensors='tf')
            umk = np.array(tokens['input_ids']).reshape(-1)
            mk = np.array(tokens['attention_mask']).reshape(-1)
            all_sent.append(umk)
            all_mask_sent.append(mk)


        return np.array(all_sent),np.array(all_mask_sent)




    def create_model(self,path_bert = 'vinai/phobert-base', num_class = 8, MAX_LEN = 20):

        phobert = TFAutoModel.from_pretrained(path_bert)
        ids = tf.keras.layers.Input(shape=(MAX_LEN), dtype=tf.int32)
        mask = tf.keras.layers.Input(shape=(MAX_LEN,), name='attention_mask', dtype='int32')
        # For transformers v4.x+: 
        embeddings = phobert(ids,attention_mask = mask)[0]
        X =tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,return_sequences=True))(embeddings)
        X =  Attention()(X)
        X = tf.keras.layers.Dense(128, activation='relu', name = 'dense1')(X)
        X = tf.keras.layers.Dropout(0.5)(X)
        y = tf.keras.layers.Dense(num_class, activation='softmax', name='outputs')(X)
        
        model = tf.keras.models.Model(inputs=[ids,mask], outputs=[y])
        model.layers[2].trainable = False
        self.model = model
      

    def create_model_test(self):
        pass

    def train(self,X_train,X_train_mask,y_true):
        self.model.fit((X_train,X_train_mask),y_true,epochs = self.epochs, batch_size = self.batch_size)
    
    
    def get_summary(self):
        self.model.summary()
    
    
    def get_predict(self,sent):
        trans ={0: 'cant_hear', 1:'intent_affirm',2: 'intent_deny_confirm', 3:'intent_number_phone',4:'provide_address',5: 'provide_code_customer', 6: 'provide_name', 7: 'this_phone', 8 :'fallback'}
        Raw = self.pre_processing(sent)
        X_test , X_test_mask = self.encoding(Raw)


        raw_pred = self.model.predict((X_test,X_test_mask))
        pred = np.argmax(raw_pred, axis = 1)
        score = raw_pred[0][pred]

        pred = trans[pred[0]]
        # print(raw_pred.shape)
        
        return pred,score

    def predict_test(self,sent):
            return 'provide_name'
    
    

    def save_weight(self,path_save, num_last_lays = 6):
        self.model.save_weights(path_save)
          
    
    def load_weight(self, path_weight):
        self.model.load_weights(path_weight)

    

