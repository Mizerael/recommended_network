import pandas as pd

from flask import render_template
from app import app
from app.source import *

sypnopsis_data = pd.read_csv(ANIME_DIR + 'anime_with_synopsis.csv'
                             , index_col='MAL_ID')
dict = {'Unknown' : 0}
sypnopsis_data['Score'] = sypnopsis_data['Score'].apply(lambda x : dict[x] if x == 'Unknown' else x).astype(float)

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/recomendations/')
def recomendations():
    top_10 = sypnopsis_data.sort_values(by= 'Score', ascending=False).head(10).values.tolist()
    return render_template("recomendations.html", titles = top_10)