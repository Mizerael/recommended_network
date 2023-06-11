from flask import Flask
from flask_bootstrap import Bootstrap
import os

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'aboba'

app = Flask(__name__)
app.config.from_object(Config)


bootstrap = Bootstrap(app)

from app import views