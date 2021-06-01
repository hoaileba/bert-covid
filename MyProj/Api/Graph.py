
import json
import random
from random import randint
path_graph = '/MyProj/DataGraph/graph.json'

class Graph:
    def __init__(self,path_action = '/home/hoaileba/PythonFlask/nlp-chatbot/MyProj/DataGraph/graph.txt'):
        self.connection = {}
        self.action_text = {}
        self.checked_action = {}
        f = open(path_action)
        for action in f:
            action = action.split("\n")[0]
            self.checked_action[action] = 0
        f.close()

    def load_check_action(self,gr):
        self.checked_action = gr

    def check_visited(self,action):
        return self.checked_action[action]
    
    def set_visited(self, action):
        self.checked_action[action] +=1

    def reset_checked(self):
        for action in self.checked_action:
            self.checked_action[action] = 0

    def check_intent(self,start_action,intent):
        if ((start_action,intent) in self.connection) == False :
            return False
        return True

    def get_checked_action(self):
        return self.checked_action

    def get_next_action(self,start_action,intent):
        return self.connection[(start_action,intent)]


    def get_text_action(self,action):
        ans  = self.action_text[action][0]
        if action == 'action_ask_search_method' and self.check_visited(action):
            ans = self.action_text['action_ask_method'][0]
            if self.check_visited('repeat_branch') :
                ans = self.action_text[action][0]
                if len(self.action_text[action]) > 1:
                    id = randint(0,1)
                    ans = self.action_text[action][2]
                return ans
            if len(self.action_text['action_ask_method']) > 1:
                    id = randint(0,1)
                    ans = self.action_text['action_ask_method'][id]

            return self.action_text['action_ask_method'][0]
        if len(self.action_text[action]) > 1:
                    id = randint(0,1)
                    ans = self.action_text[action][id]
        # if action == 'action_ask_search_method' and self.check_visited(action) == 2:
        #         return self.action_text[action][2]
        return self.action_text[action][0]

    def add_branch(self, start_action,target_action,intent):
        # key = sno/
        self.connection[(start_action,intent)] = target_action

    def save_graph(self,path):
        g = {}
        for p in self.connection:
            start_action,intent = p
            target_action = self.connection[p]
            key = start_action+'\t'+intent
            g[key] = target_action
        with open(path,'w') as f:
            json.dump(g,f)


    def load_Graph(self,path):
        with open(path,'r') as f:
            data = json.load(f)
        for id in data:
            start_action = id.split('\t')[0] 
            intent = id.split('\t')[1]
            target_action = data[id]
            self.connection[(start_action,intent)] = target_action

    def load_text(self,path):
        with open(path,'r') as f:
            data = json.load(f)
        for action in data:
            self.action_text[action] = data[action]
         




    