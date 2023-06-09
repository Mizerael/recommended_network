import pandas as pd

from flask import render_template, redirect
from app import app
from app.source import *
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField
from wtforms.validators import DataRequired
from app.recomendations import make_recomendations_with_genre, make_recomendations_with_sypnopsis

class TitleForm():
    title = StringField('title', validators=[DataRequired()])
    count = IntegerField('count', validators=[DataRequired()])

sypnopsis_data = pd.read_csv(ANIME_DIR + 'anime_with_synopsis.csv')
dict = {'Unknown' : 0}
sypnopsis_data['Score'] = sypnopsis_data['Score'].apply(lambda x : dict[x] if x == 'Unknown' else x).astype(float)

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/recomendations/')
def recomendations():
    top_10 = sypnopsis_data.sort_values(by= 'Score', ascending= False).head(10)[['Name', 'Genres', 'sypnopsis']].values.tolist()
    return render_template("recomendations.html", titles = top_10)

@app.route('/recomendations/<title>')
def content_based_recomendations(title):
    title = title.replace('_', ' ')
    recomendations = make_recomendations_with_sypnopsis(sypnopsis_data
                                                        , title)
    if recomendations is None:
        return render_template("empty.html", title= title)
    else:

        return render_template("recomendations.html"
                               , title= recomendations[0]
                               , titles= recomendations[1]
                               , count= 10)

@app.route('/recomendations/<title>/<count>')
def content_based_recomendations_with_count(title, count):
    if count is None:
        return redirect('/recomendations/title')
    title = title.replace('_', ' ')
    recomendations = make_recomendations_with_sypnopsis(sypnopsis_data
                                                        , title
                                                        , int(count))
    if recomendations is None:
        return render_template("empty.html", title= title)
    else:

        return render_template("recomendations.html"
                               , title= recomendations[0]
                               , titles= recomendations[1]
                               , count= count)
