import json
from flask import request, jsonify, Blueprint,render_template
from flask import Flask,current_app
import re
import requests
from flask.views import MethodView
from flask_restful import Resource, Api
from flask import Blueprint
from MyProj.Api import apiBp
from random import randint

from .Process import Process
from .Access_database import Database
process = Process()

# database = Database()
api = Api(apiBp)

class Init(Resource):
    def get(self):
    
        data = process.create_init()
        return jsonify({
            'sender': data['sender'],
            'text' : data['text']
        })
class Conservation(Resource):
    
    def post(self):

        req = request.json
        print('req: ',req)
        text = req['message']
        respone = process.create_respone(req)
        print('checked : ', process.graph.checked_action)
        return jsonify({
            'action': respone['action'],
            'intent' : respone['intent'],
            'text': respone['text'],
            'entities': respone['entities']
        })

api.add_resource(Conservation,'/apis/conversation')
api.add_resource(Init,'/apis/init')

