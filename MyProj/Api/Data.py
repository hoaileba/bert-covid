from flask import Flask
from MyProj import app
from flask_sqlalchemy import SQLAlchemy

# app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
db = SQLAlchemy(app)

class Conversation(db.Model):
    __tablename__ = 'conversation'
    id_convers = db.Column(db.Integer, autoincrement = True)
    sender = db.Column(db.String(30),primary_key = True)
    graph = db.Column(db.String(20000), nullable = False)
    mess  = db.relationship('Message', backref='message', lazy=True)


class Message(db.Model):
    # id = db.Column(db.Integer, )
    __tablename__ = 'message'
    id_message  = db.Column(db.Integer, primary_key=True, autoincrement = True)
    sender = db.Column(db.String(30), db.ForeignKey('conversation.sender'),nullable=False)
    action = db.Column(db.String(200), nullable=False)
    intent = db.Column(db.String(200), nullable=False)
    entities = db.Column(db.String(200), nullable=False)



db.create_all()
    # def __repr__(self):
    #     return '<User %r>' % self.username