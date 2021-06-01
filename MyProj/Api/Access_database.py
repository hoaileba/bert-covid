from random import randint
import time
from .Data import db, Conversation,Message
class Database:
        def __init__(self):
                pass
        
        def check_lich(self, address):
                x = randint(0,50)
                x = x%2 
                time.sleep(2)
                return x

        def check_by_(self,address = None, phone_num = None, ID = None):
                if address != None:
                        pass
                if phone_num != None:
                        pass
                if ID != None:
                        pass

        def check_pay(self,):
                time.sleep(3)
                rd = randint(0,3)
                return rd
          
        def check_user(self, name):
                x = randint(0,50)
                x = x%2 
                time.sleep(2)
                return x
        
        def write_Convers(self, data,graph):
                me = Conversation(sender = data['sender'], graph = graph)
                db.session.add(me)
                db.session.commit()

        def write_Message(self,data):
                # pass
                me = Message(sender = data['sender'], action = data['action'],intent = data['intent'],entities = data['entities'])
                db.session.add(me)
                db.session.commit()

        def update_graph(self, graph, sender ):
                user = Conversation.query.filter_by(sender = sender).first()
                user.graph = graph
                db.session.commit()

        def get_graph(self, sender):
                user = Conversation.query.filter_by(sender = sender).first()
                return user.graph 

        def get_last_request(self, sender):
                # pass
                data_sender = Message.query.filter_by(sender = sender).all()
                data_sender = data_sender[-1]
                # data_sender = db.session.query(Conversation).order_by(Conversation.timechat.desc()).first()
                print('data_sender: ', data_sender)
                # time.sleep(2)
                return {'intent' : data_sender.intent, 'action' : data_sender.action, 'entities':data_sender.entities,'sender':data_sender.sender}
