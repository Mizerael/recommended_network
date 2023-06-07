import pickle
import os
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


from source import *

def make_recomendations_with_cf(cf_matrix, name = ""
                                , count_recomendations = 10
                                , model_path = 'cf_model.sav'):
    if os.path.exists(model_path):
        model = pickle.load(open(model_path, 'rb'))
    else:
        cf_matrix_csr = csr_matrix(cf_matrix.values)
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(cf_matrix_csr)

        pickle.dump(model, open(model_path, 'wb'))

    if name == "":
        name = np.random.choice(cf_matrix.index.values)
    try:
        index = cf_matrix.index.get_loc(name)
        dist, ind = model.kneighbors(cf_matrix.
                                     iloc[index, :].
                                     values.reshape(1, -1)
                                     , n_neighbors= count_recomendations)
        recomendations = []
        for i, index in enumerate(ind.flatten()):
            recomendations.append([cf_matrix.index[index]
                                   , dist.flatten()[i]])
        
        return name, recomendations
    except:
        return None
    
def make_recomendations_with_genre(cbf_matrix, name = ""
                                   , count_recomendations = 10
                                   , kernels= cosine_similarity):
    if os.path.exists(CBF_GENRE_DATA) and kernels is cosine_similarity:
        similarity_matrix = np.load(CBF_GENRE_DATA)
    else:
        
        genres = cbf_matrix['Genres'].str.split(', ').astype(str)
        tfidfv = TfidfVectorizer()
        tfidf_genres = tfidfv.fit_transform(genres)
        similarity_matrix = kernels(tfidf_genres, tfidf_genres)

        if kernels is cosine_similarity:
            np.save(CBF_GENRE_DATA, similarity_matrix)

    anime_indexes = pd.Series(cbf_matrix.index, index=cbf_matrix['Name'])
    if name == "":
        name = np.random.choice(anime_indexes.index)
    try:
        similarity_scores = sorted(
            list(enumerate(similarity_matrix[anime_indexes[name]]))
            , key= lambda x: x[1]
            , reverse= True)
        similarity_scores = list(filter(lambda x: x[0] != anime_indexes[name]
                                        , similarity_scores))

        ind = similarity_scores[0: count_recomendations]
        recomendations = []
        
        for _, index in enumerate(ind):
            recomendations.append([cbf_matrix['Name'].iloc[index[0]], index[1]])

        return name, recomendations
    except:
        return None

def make_recomendations_with_sypnopsis(cbf_matrix, name = ""
                                   , count_recomendations = 10
                                   , kernels= cosine_similarity):
    if os.path.exists(CBF_SYPNOPSIS_DATA) and kernels is cosine_similarity:
        similarity_matrix = np.load(CBF_SYPNOPSIS_DATA)
    else:
        
        synopsis = cbf_matrix['sypnopsis'].str.strip(',.!?:"()').str.split(' ').astype(str)
        tfidfv = TfidfVectorizer()
        tfidf_sypnosis = tfidfv.fit_transform(synopsis)
        similarity_matrix = kernels(tfidf_sypnosis, tfidf_sypnosis)

        if kernels is cosine_similarity:
            np.save(CBF_SYPNOPSIS_DATA, similarity_matrix)

    anime_indexes = pd.Series(cbf_matrix.index, index=cbf_matrix['Name'])
    if name == "":
        name = np.random.choice(anime_indexes.index)
    try:
        similarity_scores = sorted(
            list(enumerate(similarity_matrix[anime_indexes[name]]))
            , key= lambda x: x[1]
            , reverse= True)
        similarity_scores = list(filter(lambda x: x[0] != anime_indexes[name]
                                            , similarity_scores))

        ind = similarity_scores[0: count_recomendations]
        recomendations = []

        for _, index in enumerate(ind):
            recomendations.append([cbf_matrix['Name'].iloc[index[0]], index[1]])

        return name, recomendations
    except:
        return None