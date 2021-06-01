from os import environ

DATABASE = environ.get('DATABASE')
SECRET_KEY = environ.get('SECRET_KEY')
APP_ROUTE = environ.get('APP_ROUTE')
FLASK_RUN_HOST = environ.get('FLASK_RUN_HOST')
FLASK_RUN_PORT = environ.get("FLASK_RUN_PORT")