from flask import Flask
import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'aboba'

app = Flask(__name__)
app.config.from_object(Config)

from app import views