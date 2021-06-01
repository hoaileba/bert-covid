
from flask import Blueprint, flash, render_template, request, session, abort, \
                  redirect, url_for
import requests
import os
# from app import app

appbp = Blueprint('auth', __name__, url_prefix='')

@appbp.route("/chat")
def hello():
    return render_template('chat1.html')