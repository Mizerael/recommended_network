import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

from flask import render_template, redirect, flash
from app import app
from app.source import *
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField
from wtforms.validators import DataRequired
from app.recomendations import make_recomendations_with_sypnopsis, vectorization

class TitleForm(FlaskForm):
    title = StringField('title', validators=[DataRequired()])
    count = IntegerField('count', validators=[DataRequired()], default=10)
    submit = SubmitField('Search')

sypnopsis_data = pd.read_csv(ANIME_DIR + 'anime_with_synopsis.csv')
dict = {'Unknown' : 0}
sypnopsis_data['Score'] = sypnopsis_data['Score'] \
                            .apply(lambda x : dict[x] if x == 'Unknown' else x)\
                            .astype(float)

images_links = pd.read_csv('app/static/images_links.csv', index_col='id')

if os.path.exists(CBF_SYPNOPSIS_DATA):
    sypnopsis_similarity = np.load(CBF_SYPNOPSIS_DATA)
else:
    sypnopsis = sypnopsis_data['sypnopsis'].str.strip(',.!?:"()') \
                                           .str.split(' ') \
                                           .astype(str)
    sypnopsis_similarity = vectorization(sypnopsis
                                         , CBF_SYPNOPSIS_DATA
                                         , cosine_similarity)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = TitleForm()
    if form.submit():
        if sypnopsis_data[sypnopsis_data['Name'] == form.title.data].empty:
            if form.title.data != None:
                    flash(f'Anime not found {form.title.data}') 
            return render_template("home.html", form= form)
        else:
            return redirect(f'''/{form.title.data
                                            .replace(" ", "_")
                                            .lower()}/{form.count.data}''')
            
    
    return render_template("home.html", form= form)

@app.route('/anime_top/')
def recomendations():
    top_10 = sypnopsis_data.sort_values(by= 'Score', ascending= False) \
                           .head(10)[['Name', 'Genres', 'sypnopsis']] \
                           .values.tolist()
    images = []
    for x in top_10:
        link = images_links[images_links["title"] == x[0]]["image_url"].values \
                                                                       .tolist()
        if link == []:
            link = ''
        else:
            link = link[0]
        images.append(link)
    top_10 = zip(top_10, images)
    return render_template("recomendations.html", titles = top_10)

@app.route('/random')
def random_title():
    title = np.random.choice(sypnopsis_data['Name']).replace(' ', '_').lower()
    return redirect(f'/{title}')

@app.route('/<title>')
def content_based_recomendations(title):
    title = title.replace('_', ' ').lower()
    recomendations = make_recomendations_with_sypnopsis(sypnopsis_data
                                                        , title
                                                        , similarity_matrix= 
                                                          sypnopsis_similarity)
    if recomendations is None:
        return render_template("empty.html", title= title)
    else:
        images = []
        for x in recomendations[1]:
            link = images_links[images_links["title"] == x[0]]["image_url"] \
                     .values \
                     .tolist()
            if link == []:
                link = ''
            else:
                link = link[0]
            images.append(link)
        anime_titles = zip(recomendations[1], images)

        link = images_links[images_links["title"]
                              .str.lower() == title]['image_url'].values \
                                                                 .tolist()
        if link == []:
            link = ''
        else:
           link = link[0]
        
        return render_template("recomendations.html"
                               , title= [recomendations[0], link]
                               , titles= anime_titles
                               , count= 10)

@app.route('/<title>/<count>')
def content_based_recomendations_with_count(title, count):
    if count == '' or count == '10':
        return redirect(f'/{title}')
    title = title.replace('_', ' ')
    recomendations = make_recomendations_with_sypnopsis(sypnopsis_data
                                                        , title
                                                        , int(count)
                                                        , similarity_matrix=
                                                          sypnopsis_similarity)
    if recomendations is None:
        return render_template("empty.html", title= title)
    else:
        images = []
        for x in recomendations[1]:
            link = images_links[images_links["title"] == x[0]]["image_url"] \
                     .values.tolist()
            if link == []:
                link = ''
            else:
                link = link[0]
            images.append(link)
        anime_titles = zip(recomendations[1], images)
        
        link = images_links[images_links["title"]
                              .str.lower() == title]['image_url'].values \
                                                                 .tolist()
        if link == []:
            link = ''
        else:
           link = link[0]

        return render_template("recomendations.html"
                                , title= [recomendations[0], link]
                                , titles= anime_titles
                                , count= count)