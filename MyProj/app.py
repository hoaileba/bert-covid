import json, requests
from io import BytesIO
from flask import Flask, request, jsonify,url_for, Blueprint,render_template
import logging
from . import create_app,socketio
from flask_socketio import SocketIO, emit
app = create_app()
socketio = SocketIO(app,cors_allowed_origins="*")
# from .Api.Fake import Database
# database = Database()

import json
# @app.route('/chat')
# def chat():
#     return render_template('chat1.html')

    
@socketio.on('Bot', namespace = '/chat')
def Bot(message):

    req  = message
    headers={"Content-Type":"application/json"}
    response = requests.post("http://localhost:5005/apis/conversation", data = json.dumps(message),headers  =headers)

    x = response.json()
    x['sender'] = req['sender']

    action = x['action']
    
    emit('message_bot', {'message':x['text'], 'bot_action': x['action']})
    if action == 'action_please_wait':
            response = requests.post("http://localhost:5005/apis/conversation", data = json.dumps(message),headers  =headers)
            x = response.json()
            x['sender'] = req['sender']
            emit('message_bot', {'message':x['text'], 'bot_action': x['action']})

    #  if action is please wait for checking ->> request.post 1 more time then emit again
    


@socketio.on('initDialogue', namespace = '/chat')
def Init(message):
    
    sender = requests.get("http://localhost:5005/apis/init")
    x = sender.json()
    print("X:",x)
    emit('status', {'sender_id':x['sender'], 'text': x['text']})

