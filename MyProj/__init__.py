
import os
import time
import json
from flask import Flask,render_template
from .Pages.page import appbp
# from flask import Blueprint
# from flask_socketio import SocketIO, emit  ,send 
# from . import db
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO 
socketio = SocketIO()



# main_blue = Blueprint('main', __name__)

def create_app(test_config=None):
    # create and configure the app
    instance_path = "MyProj/models"
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    


    app = Flask(__name__, instance_relative_config=True)
    # app.register_blueprint(main_blue)
    app.config.from_mapping(
        SECRET_KEY='dev',
        SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'app.db')
    )
    # app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
    # db = SQLAlchemy(app)
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    app.register_blueprint(appbp)
    # db.init_app(app)
    # @app.route('/hello')
    # def hello():
    #     return render_template('chat1.html')
    # from .api import main as main_blueprint
    # app.register_blueprint(main_blueprint)
    socketio.init_app(app)
    return app


app = create_app()

# from . import api