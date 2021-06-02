from .Model_Intent import Model_Cls
from .Graph import Graph
from .Access_database import Database
import json
from datetime import datetime
from random import randint

PATH_GRAPH = 'MyProj/DataGraph/graph_covid.json'
PATH_INTENT = 'MyProj/weight_model/weight_intent.h5'
PATH_TEXT = 'MyProj/DataGraph/text_action _covid.json'
list_method_action = ['action_ask_address_contract','action_ask_number','action_ask_ID']
list_method_text = ['địa chỉ', 'số điện thoại','mã khách hàng']
THRESHOLD = 0.5

database = Database()

model_intent = Model_Cls()
model_intent.create_model(path_bert='/home/hoaileba/PythonFlask/nlp-chatbot/MyProj/weight_model/pho_pretrained')
model_intent.load_weight(PATH_INTENT)

graph = Graph()
graph.load_Graph(PATH_GRAPH)
graph.load_text(PATH_TEXT)


class Process_Case:
        def __init__(self,db = database,graph = graph):
                self.db = db
                self.graph = graph
        def processing_text(self,text,intent,action):
                pass


class Process:
        def __init__(self,model_intent = model_intent,graph = graph, db = database):
                self.model_intent = model_intent
                self.graph = graph
                self.db = db
                
        def get_pred_intent(self,text):
                
                return self.model_intent.get_predict(text)
        

        def create_init(self):
                now = datetime.now()
                
                # print("now =", now)
                ra = randint(0,100000)
                dt_string = now.strftime("%d/%m/%Y/%H/%M/%S")+str(ra)
                text =  (self.graph.get_next_action('action_start','begin'))
                text = self.graph.get_text_action(text)
                self.graph.reset_checked()
                self.graph.set_visited('action_ask_begin')
                gr = self.graph.get_checked_action()
                gr = json.dumps(gr)
                print(text)
                # gr = 
                data = {
                        'sender':dt_string,
                        'action': 'action_ask_begin',
                        'intent' : 'begin' ,
                        'text' : "",
                        'entities': "",
                        

                }
                self.db.write_Convers(data,graph = gr)
                self.db.write_Message(data)
                
                # self.db.update_graph(gr,dt_string)
                return {
                        "text":text,
                        'sender':dt_string
                }
                
        def create_respone(self,request):
                
                text = request['message']
                sender = request['sender']

# get raw predict intent, score and entities
                intent, score  = self.get_pred_intent(text)
                entities = [{}]


                gr = self.db.get_graph(sender=  sender)
                gr = eval(gr)

                self.graph.load_check_action(gr)
                # print('raw_predict_intent: ', intent,' - score: ',score)
                
# get last action 
                previous_request = self.db.get_last_request(sender)
                previous_action = "action_ask_begin"
                previous_intent = previous_request['intent']

                
                current_action = (self.graph.get_next_action(previous_action,intent))
                final_action = current_action
                
                text = self.graph.get_text_action(final_action)
                self.graph.set_visited(final_action)
                print('pre_action =========== ',previous_action,'    - score: ',score)
                print('intent =============== ', intent)
                print("next_action ========== ", final_action)
                data = {
                        'sender':sender,
                        'action': final_action,
                        'intent' : intent ,
                        'text' : text,
                        'entities': ""
                }
                gr = self.graph.get_checked_action()
                gr = json.dumps(gr)
                self.db.write_Message(data)
                self.db.update_graph(sender = sender, graph = gr)
                print(data)
                print('-------------------------------------END-------------------------------')
                return data


