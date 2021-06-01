import numpy as np
import tensorflow as tf
import pandas as pd
import os
import json
import re
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate
from tensorflow.keras.layers import Layer
from transformers import TFAutoModel, PhobertTokenizer,AutoTokenizer
import tensorflow_addons as tfa


TRAINING_MODE = 'training'
PREDICTION_MODE = 'prediction'


class CRFLayer(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 chain_initializer='orthogonal',
                 use_boundary=True,
                 boundary_initializer='zeros',
                 use_kernel=True,
                 **kwargs):
            super().__init__(**kwargs)
            self.crf = tfa.layers.CRF(
                units,
                chain_initializer=chain_initializer,
                use_boundary=use_boundary,
                boundary_initializer=boundary_initializer,
                use_kernel=use_kernel,
                **kwargs)
            self.units = units
            self.chain_kernel = self.crf.chain_kernel
            # record sequence length to compute loss
            self.sequence_length = None
            self.mask = None
            self.mode = TRAINING_MODE

    def call(self, inputs, training=None, mask=None):
            """Forward pass.
            Args:
                inputs: A [batch_size, max_seq_len, depth] tensor, inputs of CRF layer
                mask: A [batch_size, max_seq_len] boolean tensor, used to compulte sequence length in CRF layer
            Returns:
                potentials: A [batch_size, max_seq_len, units] tensor in train phase.
                sequence: A [batch_size, max_seq_len, units] tensor of decoded sequence in predict phase.
            """
            self.mode = TRAINING_MODE if training is True else PREDICTION_MODE
            sequence, potentials, sequence_length, transitions = self.crf(inputs, mask=mask)
            # sequence_length is computed in both train and predict phase
            self.sequence_length = sequence_length
            # save mask, which is needed to compute accuracy
            self.mask = mask

            sequence = tf.cast(tf.one_hot(sequence, self.units), dtype=self.dtype)
            return tf.keras.backend.in_train_phase(potentials, sequence)

    def accuracy(self, y_true, y_pred):
                if len(tf.keras.backend.int_shape(y_true)) == 3:
                        y_true = tf.argmax(y_true, axis=-1)
                if self.mode == PREDICTION_MODE:
                        y_pred = tf.argmax(y_pred, axis=-1)
                else:
                        y_pred, _ = tfa.text.crf_decode(y_pred, self.chain_kernel, self.sequence_length)
                y_pred = tf.cast(y_pred, dtype=y_true.dtype)
                equals = tf.cast(tf.equal(y_true, y_pred), y_true.dtype)
                if self.mask is not None:
                        mask = tf.cast(self.mask, y_true.dtype)
                        equals = equals * mask
                        return tf.reduce_sum(equals) / tf.reduce_sum(mask)
                return tf.reduce_mean(equals)

    def neg_log_likelihood(self, y_true, y_pred):
                # print(y_true,y_pred)
                # loss = tf.keras.losses.SparseCategoricalCrossentropy(y_true,y_pred)
                log_likelihood, _ = tfa.text.crf_log_likelihood(y_pred, y_true, self.sequence_length, self.chain_kernel)
                return tf.reduce_max(-log_likelihood)

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

class Model_NER:
    def __init__(self,batch_size = 64, epochs = 10):
                self.batch_size = batch_size
                self.epochs = epochs
                self.tokenizer =  AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

    def encoding(self,sent,max_length = 20):
        
                all_sent = []
                all_mask_sent = []
                # print(sent)
                for line in sent:
                # print(line.shape)
                # l = tokenizer.encode(line)
                # l = rdrsegmenter.tokenize(line)
                        tokens = self.tokenizer.encode_plus(line, max_length=max_length, padding= 'max_length',
                                                        truncation=True, 
                                                        add_special_tokens=True, return_attention_mask=True,
                                                        return_token_type_ids=False, return_tensor = 'tf')
                        umk = np.array(tokens['input_ids']).reshape(-1)
                        mk = np.array(tokens['attention_mask']).reshape(-1)
                        all_sent.append(umk)
                        all_mask_sent.append(mk)
                # print(umk)
                # time.sleep(2)
                # print(np.array(all_sent).shape)
                # print(all_sent[0])
                # print(all_mask_sent[0])
                # all_sent = padding(all_sent,max_length=max_length)
                # all_mask_sent = padding(all_mask_sent,max_length=max_length)
                return np.array(all_sent),np.array(all_mask_sent)

    def create_model(self,path_bert = 'vinai/phobert-base', num_class = 6, MAX_LEN = 20):

                phobert = TFAutoModel.from_pretrained(path_bert)
                ids = tf.keras.layers.Input(shape=(MAX_LEN), dtype='int32')
                # sequence_mask = tf.keras.layers.Lambda(lambda x: tf.greater(x, 0))(ids)
                mask_ = tf.keras.layers.Input(shape=(MAX_LEN,), name='attention_mask', dtype='int32')
                # For transformers v4.x+: 

                embeddings = phobert(ids,attention_mask = mask_)[0]
                # print(embeddings[:,0,:])

                # Y = tf.keras.layers.TimeDistributed(Dense(5,activation='relu'))(embeddings)
                X =tf.keras.layers.Bidirectional( tf.keras.layers.LSTM(128,return_sequences=True,dropout=0.5))(embeddings)
                X = tf.keras.layers.TimeDistributed(Dense(128,activation='relu'))(X)
                # X = tf.keras.layers.Dense(256, activation='relu', name='outputs')(X)

                crf = CRFLayer(9)
                Y = crf(X,mask =mask_)
                model = tf.keras.models.Model(inputs=[ids,mask_], outputs=[Y])
                model.summary()
                model.layers[2].trainable = False
                self.model = model
        

    def train(self,X_train,X_train_mask,y_true):
                self.model.fit((X_train,X_train_mask),y_true,epochs = self.epochs, batch_size = self.batch_size)
    
    
    def get_summary(self):
                self.model.summary()
    
    
    def get_predict(self,sent):
                trans_off = {0:'O',1:'B-PER',2:'B-PRV',3:'B-DIS',4:'B-STR',5:'I-PER',6:'I-PRV',7:'I-DIS',8:'I-STR'}
                # Raw = self.pre_processing(sent)
                print(sent)
                sent = sent[0].split(' ')
                sent =[sent]
                X_test , X_test_mask = self.encoding(sent)
                entities = []
                name = ''
                street = ''
                dis = ''
                province  = ''

                raw_pred = self.model.predict((X_test,X_test_mask))
                pred = np.argmax(raw_pred, axis = 2)
                # pred = np.argmax(pred,axis= 2)
                print(pred[0])
                data = pred[0][1:]
                tmp = ""
                index = []
                # for id,t in enumerate(sent[0]):
                #         print(t,trans_off[data[id]])
                sent[0].append(0)
                for id,t in enumerate(sent[0]):
                        # print(t,trans_off[data[id]])
                        if data[id]!=0:
                                tmp+= sent[0][id]+' '
                                index.append(id)
                        elif index != [] :
                                # print(tmp)
                                if trans_off[data[index[0]]] == 'B-PER':
                                        name = tmp.strip()
                                        st = index[0]
                                        ed = index[-1]
                                        entities.append({'start':st,'end':ed,'value':name,'type':'NAME','method':'model'})
                                if trans_off[data[index[0]]] == 'B-STR':
                                        street = tmp.strip()
                                        st = index[0]
                                        ed = index[-1]
                                        entities.append({'start':st,'end':ed,'value':street,'type':'STREET','method':'model'})
                                if trans_off[data[index[0]]] == 'B-DIS':
                                        dis = tmp.strip()
                                        st = index[0]
                                        ed = index[-1]
                                        entities.append({'start':st,'end':ed,'value':dis,'type':'DIS','method':'model'})
                                if trans_off[data[index[0]]] == 'B-PRV':
                                        province = tmp.strip()
                                        st = index[0]
                                        ed = index[-1]
                                        entities.append({'start':st,'end':ed,'value':province,'type':'PRV','method':'model'})
                                tmp = ''
                                index = []
                
                if entities == []:
                        entities = [{}]
                else :
                        add = ''
                        Min = 30
                        Max = 0
                        for e in entities :
                                if e['type'] == 'STREET':
                                        add+=e['value'] + ' '
                                        Min = min(e['start'],Min)
                                        Max = max(e['end'],Max)
                                if e['type'] == 'DIS':
                                        Min = min(e['start'],Min)
                                        Max = max(e['end'],Max)
                                        add+=e['value'] + ' '
                                if e['type'] == 'PRV':
                                        Min = min(e['start'],Min)
                                        Max = max(e['end'],Max)
                                        add+=e['value'] + ' '
                        if add !='':
                                entities.append({'start':Min,'end':Max,'value':add, 'type':'address','method':'model'})
                        else :
                                entities.append({})

                return entities
                        # if trans_off[data[id]] == 'B'
                
                # score = raw_pred[0][pred]

                # pred = trans[pred[0]]
                # print(raw_pred.shape)
                
                # return pred,score

    def save_weight(self,path_save, num_last_lays = 6):
            self.model.save_weights(path_save)
          
    
    def load_weight(self, path_weight):
            self.model.load_weights(path_weight)
    
    def create_model_test(self):
            pass

    def predict_phone(self,string ):
            
            reg = re.compile("\d{10}")
            x = reg.search(string)
            entities = [{}]
            if x != None:
                   
                    st = x.start()
                    ed = x.end()
                    value = x.group()
                    entities = [{'start':st, 'end': ed, 'value':value,'type':'phone_number','method':'regex'}]
            return entities

    def predict_code_cus(self,text):
            regex = re.compile('(p|P)\w{5}\d{7}')
            x = regex.search(text)
            entities  = [{}]
            if x != None:
                    st = x.start()
                    ed = x.end()
                    value = x.group()
                    entities = [{'start':st, 'end': ed, 'value':value,'type':'ID','method':'regex'}]
            return entities

    def get_predict_test(self,text):
            return [{}]

# model_ner = Model_NER()
# model_ner.create_model()
# model_ner.load_weight(PATH_ENT)

# print(model_ner.get_predict(['phường trung văn, quận hà đông, thành phố hà nội']))
