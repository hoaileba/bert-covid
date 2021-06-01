from .Model_Intent import Model_Cls
from .Model_NER import Model_NER
from .Graph import Graph
from .Access_database import Database
import json
from datetime import datetime
from random import randint

PATH_GRAPH = 'MyProj/DataGraph/graph_EVN_paycheck.json'
PATH_INTENT = 'MyProj/weight_model/weight.h5'
PATH_ENT = '/home/hoaileba/PythonFlask/nlp-chatbot/MyProj/weight_model/weight_NER2-4.h5'
PATH_TEXT = 'MyProj/DataGraph/text_action_EVN2.json'
list_method_action = ['action_ask_address_contract','action_ask_number','action_ask_ID']
list_method_text = ['địa chỉ', 'số điện thoại','mã khách hàng']
THRESHOLD = 0.5

database = Database()

model_intent = Model_Cls()
model_intent.create_model(path_bert='/home/hoaileba/PythonFlask/nlp-chatbot/MyProj/weight_model/pho_pretrained')
model_intent.load_weight(PATH_INTENT)

model_ner = Model_NER()
model_ner.create_model(path_bert='/home/hoaileba/PythonFlask/nlp-chatbot/MyProj/weight_model/pho_pretrained')
model_ner.load_weight(PATH_ENT)

graph = Graph()
graph.load_Graph(PATH_GRAPH)
graph.load_text(PATH_TEXT)


class Process_Case:
        def __init__(self,db = database,graph = graph):
                self.db = db
                self.graph = graph
                self.province22 =  ['thanh hóa', 'nghệ an', 'hà tĩnh', 'quảng bình', \
                        'quảng trị', 'thừa thiên-huế','thừa thiên-huế','huế', 'đà nẵng', 'quảng nam', 'quảng ngãi',\
                                 'bình định', 'phú yên', 'khánh hòa', 'ninh thuận', 'bình thuận', \
                                         'kon tum', 'gia lai', 'đắc lắc', 'đắc nông', 'lâm đồng']
                self.all_province = ['an giang', 'bà rịa - vũng tàu', 'vũng tàu', 'bắc giang', 'bắc kạn', 'bạc liêu', 'bắc ninh'\
                        , 'bến tre', 'bình định', 'bình dương', 'bình phước', 'bình thuận', 'cà mau', 'cao bằng', 'đắk lắk', 'đắk nông'\
                                , 'điện biên', 'đồng nai', 'đồng tháp', 'gia lai', 'hà giang', 'hà nam', 'hà tĩnh', 'hải dương', 'hậu giang'\
                                        , 'hòa bình', 'hưng yên', 'khánh hòa', 'kiên giang', 'kon tum', 'lai châu', 'lâm đồng', 'lạng sơn',\
                                                 'lào cai', 'long an', 'nam định', 'nghệ an', 'ninh bình', 'ninh thuận', 'phú thọ', 'quảng bình',\
                                                          'quảng nam', 'quảng ngãi', 'quảng ninh', 'quảng trị', 'sóc trăng', 'sơn la', 'tây ninh',\
                                                                   'thái bình', 'thái nguyên', 'thanh hóa', 'thừa thiên huế', 'huế', 'tiền giang',\
                                                                            'trà vinh', 'tuyên quang', 'vĩnh long', 'vĩnh phúc', 'yên bái', 'phú yên',\
                                                                                     'cần thơ', 'đà nẵng', 'hải phòng', 'hà nội', 'tp hcm']


                # pass

        def check_exist_province(self,entities):
                province = {}
                for e in entities :
                        if e['type'] == 'PRV':
                                province = e
                                break
                if province == {}:
                        return 0
                string = province['value']
                string = string.strip()
                print(string)
                # print(self.province22[0])
                # if string == self.province22:
                        # print('ýey')
                if string in self.province22:
                        return 2
                elif (string in self.all_province) == False:
                        return 0
                return 1

        def process_by_score(self,intent,score):
                if score < THRESHOLD:
                        return 'fallback'
                return intent
        
        def process_by_regex(self,phone,code,intent):
                if code != [{}]:
                        intent = 'provide_code_customer'
                elif phone != [{}]:
                        intent = 'intent_number_phone'
                return intent

        def check_exist_path(self,previous_action, intent,entities):

                intent = self.check_province(intent,entities,previous_action)
                print(intent)
                if self.graph.check_intent(previous_action,intent) == False:
                        return 'fallback'
                return intent
        
        def check_entities(self,intent,entities,previous_action):
                if intent == "intent_provide_name" :
                        en = [{}]
                        if entities == [{}]:
                                return "no_entities"
                        for e in entities:
                                if e['type'] == 'NAME':
                                        en = e
                                        break
                        if en == [{}]:
                                return "no_entities"
                if intent == "intent_provide_address":
                        en = [{}]
                        if entities == [{}]:
                                return "no_entities"
                        # if entities 
                        for e in entities:
                                if e['type'] == 'address':
                                        en = e
                                        break
                        if en == [{}]:
                                return "no_entities"
                if intent == 'provide_address' and (previous_action != 'action_ask_search_method') and previous_action!= 'action_seach_again':
                        en = [{}]
                        if entities == [{}]:
                                return "no_entities"
                        for e in entities:
                                if e['type'] == 'address':
                                        en = e
                                        break
                        if en == [{}]:
                                return "no_entities"
                if intent == 'provide_name' :
                        en = [{}]
                        if entities == [{}]:
                                return "no_entities"
                        for e in entities:
                                if e['type'] == 'NAME':
                                        en = e
                                        break
                        if en == [{}]:
                                return "no_entities"
                if intent == 'intent_number_phone' and entities == [{}] and (previous_action != 'action_ask_search_method') and previous_action!= 'action_seach_again':
                        return 'no_entities'
                if intent == 'provide_code_customer' and entities == [{}] and (previous_action != 'action_ask_search_method') and previous_action!= 'action_seach_again':
                        return 'no_entities'
                # if intent == ''
                return intent
        
        def check_fallback_2nd(self,intent,previous_intent):
                if (previous_intent =='no_entities' or previous_intent == 'cant_hear' or previous_intent == 'fallback' )and (intent =='no_entities' ) == True:
                        intent = intent+'_1'
                else :  
                        intent = intent

                if (previous_intent =='fallback' or previous_intent == 'cant_hear')and (intent =='fallback' or intent == 'cant_hear') == True:
                        intent = intent+'_1'
                else :  
                        intent = intent
                return intent
        def check_visited_(self,previous_action,action, intent):
                if self.graph.check_visited(action) >=1 and( previous_action == 'action_ask_search_method' or previous_action == 'action_ask_method' ):
                        return 'fallback'
                return intent
                
        def check_num_in_repbranch(self, intent, current_action,previous_action ):
                if current_action == 'repeat_branch':
                        self.graph.set_visited(current_action)
                if current_action == 'repeat_branch' and self.graph.check_visited(current_action) == 1 :
                        return 'repeat_1'
                if current_action == 'repeat_branch' and self.graph.check_visited(current_action) == 2:
                        return 'repeat_2' 

                return  intent
                # pass

        def check_province(self, intent, entities , previous_action):
                print('checking')
                if intent == 'provide_address' and (previous_action == 'action_ask_province' or previous_action == 'action_ask_province_again') :
                        is_exist = self.check_exist_province(entities)
                        print('is_exist: ', is_exist)
                        # is_exist = 2
                        if is_exist == 0 :
                                return 'provide_address+check_no_province' 
                        else :
                                if is_exist == 1:
                                        return "provide_address+check_yes_province+check_no_22"
                                else:
                                        return "provide_address+check_yes_province+check_yes_22"
                return intent


        def check_possible_method(self, intent, previous_action,previous_intent):
                if self.graph.checked_action['repeat_branch'] !=0 and (previous_action =='action_ask_search_method'\
                                or previous_action == 'action_ask_method' or previous_action == 'action_seach_again'):
                        current_action = self.graph.get_next_action(previous_action,intent)
                        # if intent == ''
                        if intent == 'intent_number_phone' and (previous_action =='action_ask_search_method'\
                                or previous_action == 'action_ask_method'or previous_action == 'action_seach_again') \
                                and self.graph.check_visited(current_action) >=1:
                                intent =  'fallback'
                                return self.check_fallback_2nd(intent,previous_intent)

                        if intent == 'provide_code_customer' and (previous_action =='action_ask_search_method'\
                                or previous_action == 'action_ask_method'or previous_action == 'action_seach_again') \
                                and self.graph.check_visited(current_action) >=1:
                                intent =  'fallback'
                                return self.check_fallback_2nd(intent,previous_intent)

                        if intent == 'provide_address' and (previous_action =='action_ask_search_method'\
                                or previous_action == 'action_ask_method'or previous_action == 'action_seach_again') \
                                and self.graph.check_visited(current_action) >=1:
                                intent = 'fallback'
                                return self.check_fallback_2nd(intent,previous_intent)
                return intent


        def check_exist_database(self,action,entities,):

                if action == 'action_check_power':
                        result = self.db.check_lich(entities)
                        if result == 0:
                                final_intent = 'check_no'
                        else :
                                final_intent = 'check_yes'

                if action == 'action_check_name':
                        result = self.db.check_lich(entities)
                        if result == 0:
                                final_intent = 'check_no'
                        else :
                                final_intent = 'check_yes'

                # if action == 
                        # final_action = (self.graph.get_next_action(action,final_intent))
        def processing_text(self,text,intent,action,entities,previous_entities = None):
                
                if (action == 'action_ask_number' or action == 'action_ask_number_again') == True\
                        and intent == 'no_entities':
                        text = 'Vui lòng đọc lại số điện thoại'
                if (action == 'action_ask_ID' or action == 'action_ask_ID_again') == True\
                        and intent == 'no_entities':
                        text = 'Vui lòng đọc lại số mã khách hàng'

                # if action == 'action_confirm_number' and (intent == 'fallback') 

                if action == 'action_not_support':
                        province = ''
                        en = [{}]
                        for e in entities:
                                if e['type'] == 'PRV':
                                        province = e['value']
                                        en = e
                                        break
                        if en != [{}]:
                                province = en['value']
                        text = text.format(province = province)

                if action == 'action_confirm_address' or action == 'action_confirm_address_again':
                        address = ''
                        en = [{}]
                        for e in entities:
                                if e['type'] == 'address':
                                        en = e
                                        break
                        if en != [{}]:
                                address = en['value']
                        text = text.format(address = address)
                if (action == 'action_confirm_ID' or action == 'action_confirm_ID_again') :
                        ID = ''
                        en = [{}]
                        for e in entities:
                                if e['type'] == 'ID':
                                        en = e
                                        break
                        if en != [{}]:
                                ID = en['value']
                        text = text.format(code = ID)
                if action == 'action_confirm_name' or action == 'action_confirm_name_again':
                        name = ''
                        en = [{}]
                        for e in entities:
                                if e['type'] == 'NAME':
                                        en = e
                                        break
                        if en != [{}]:
                                name = en['value']
                        text = text.format(name = name)
                if (action == 'action_confirm_number_again' or action == 'action_confirm_number') :
                        phone = ''
                        en = [{}]
                        for e in entities:
                                if e['type'] == 'phone_number':
                                        en = e
                                        break
                        # print("e: ",e)
                        if en != [{}]:
                                phone = en['value']
                        text = text.format(phone_number = phone)
                if action == 'action_not_support':
                        province = ''
                        if entities != [{}]:
                                province = entities[0]['value']
                        text = text.format(phone_number = province)
                id = []
                if action =='action_seach_again' or (action == 'action_ask_search_method' and self.graph.check_visited ):
                        for i,method in enumerate(list_method_action):
                                if self.graph.check_visited(method) == 0:
                                        id.append(i)
                        text = text.format(method_1 = list_method_text[id[0]],method_2 = list_method_text[id[1]])
                
                
                
                return text



pc = Process_Case()



class Process:
        def __init__(self,model_intent = model_intent, model_ner = model_ner,graph = graph, db = database,process_case = pc):
                self.model_intent = model_intent
                self.model_ner = model_ner
                self.graph = graph
                self.db = db
                self.process_case = pc
                
        def get_pred_intent(self,text):
                
                return self.model_intent.get_predict(text)
        
        def get_pred_entities(self,text):

                text = [text]
                return self.model_ner.get_predict(text)

        def create_init(self):
                now = datetime.now()
                
                # print("now =", now)
                ra = randint(0,100000)
                dt_string = now.strftime("%d/%m/%Y/%H/%M/%S")+str(ra)
                text =  (self.graph.get_next_action('action_start','begin'))
                text = self.graph.get_text_action(text)
                self.graph.reset_checked()
                self.graph.set_visited('action_ask_province')
                gr = self.graph.get_checked_action()
                gr = json.dumps(gr)
                print(text)
                # gr = 
                data = {
                        'sender':dt_string,
                        'action': 'action_ask_province',
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

                code = self.model_ner.predict_code_cus(text)
                phone = self.model_ner.predict_phone(text)

                gr = self.db.get_graph(sender=  sender)
                gr = eval(gr)

                self.graph.load_check_action(gr)
                print('raw_predict_intent: ', intent,' - score: ',score)
                
# get last action 
                previous_request = self.db.get_last_request(sender)
                previous_action = previous_request['action']
                previous_intent = previous_request['intent']
# handle fallback by score -------------- filter 1
                if previous_action == 'action_please_wait':
                        list_intent = ['over_3_day','not_coming','have_paycheck','not_exist']
                        intent = list_intent[self.db.check_pay()]
                        final_action = self.graph.get_next_action(previous_action,intent)
                        text = self.graph.get_text_action(final_action)
                        data = {
                                'sender':sender,
                                'action': final_action,
                                'intent' : intent ,
                                'text' : text,
                                'entities': "[{}]"
                        }
                        gr = self.graph.get_checked_action()
                        gr = json.dumps(gr)
                        self.db.write_Message(data)
                        self.db.update_graph(sender = sender, graph = gr)
                        return data
                        
                
# ----------------------------------process by score ----------------------------------------------------

                intent = self.process_case.process_by_score(intent,score)
                intent = self.process_case.process_by_regex(phone,code,intent)
                print("process_by_score: ", intent)
                if intent == 'intent_number_phone' : 
                        entities = self.model_ner.predict_phone(text)
                if previous_action == 'action_ask_ID' or previous_action == 'action_ask_ID_again' or previous_action == 'action_confirm_ID' or previous_action == 'action_confirm_ID_again':
                        e = self.model_ner.predict_code_cus(text)
                        print(e)
                        if e != [{}]:
                                intent = 'provide_code_customer'

                if intent == 'provide_code_customer':
                        entities = self.model_ner.predict_code_cus(text)
                
                
#check exist intent  ---------------- filter 2
                if intent == 'provide_name' or intent == 'provide_address':
                        entities = self.get_pred_entities(text)
                        print('entities: ', entities)
                intent = self.process_case.check_exist_path(previous_action, intent,entities   )
                
                print('check_exist_path : ', intent)

                
# check 2nd fallback ---------------- filter 3 ---------------------------------------------------------------

                intent = self.process_case.check_entities(intent,entities,previous_action)
                intent = self.process_case.check_exist_path(previous_action, intent,entities)
                print('check_entities: ', intent)

#check entities  ------------------- filter 4-----------------------------------------------------------------
                
                intent = self.process_case.check_fallback_2nd(intent,previous_intent)
                
                print("check_fallback_2nd: ", intent)

#check reapeat branch --------------- filter 5 ---------------------------------------------------------------
                current_action = (self.graph.get_next_action(previous_action,intent))
                print('current_action: ', current_action)
# ------------------------------------------------------------------------------------------------------------

                intent = self.process_case.check_num_in_repbranch(intent,current_action,previous_action)
                print('check_repeat_branch: ', intent)

# ------------------------------------------------------------------------------------------------------------

                intent = self.process_case.check_possible_method(intent,previous_action,previous_intent)
                print('check_possible_method: ', intent)
# ------------------------------------------------------------------------------------------------------------
                
                # intent = self.process_case.check_exist_path(previous_action, intent,entities)
# check name and address exists in database or not
                if current_action == 'repeat_branch':
                        previous_action = current_action
                
                current_action = (self.graph.get_next_action(previous_action,intent))
                final_action = current_action
                
                if (final_action == 'action_confirm_number' or final_action == 'action_confirm_name'\
                         or final_action == 'action_confirm_address' or final_action == 'action_confirm_ID') and intent == 'fallback':
                         
                         entities = eval(previous_request['entities'])
                         print('pre_entities: ', entities)


                print('final_action: ', final_action)
                
                text = self.graph.get_text_action(final_action)
                print("pre_text: ", text)
                self.graph.set_visited(final_action)
                text = self.process_case.processing_text(text,intent,final_action,entities)
                print('after :',text)
                entities = str(entities)
                print('final_intent: ',intent)
                # print(entities)
                data = {
                        'sender':sender,
                        'action': final_action,
                        'intent' : intent ,
                        'text' : text,
                        'entities': entities
                }
                gr = self.graph.get_checked_action()
                gr = json.dumps(gr)
                self.db.write_Message(data)
                self.db.update_graph(sender = sender, graph = gr)
                print(data)
                print('-------------------------------------END-------------------------------')
                return data


