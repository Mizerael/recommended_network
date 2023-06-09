import pickle
import os
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


from app.source import *

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
    
def get_scores(matrix, indexes, name, is_sorted= True):
    similarity_scores = list(enumerate(matrix[indexes[name]]))
    if is_sorted:
        similarity_scores = sorted(similarity_scores
                                   , key= lambda x: x[1], reverse= True)
    return list(filter(lambda x: x[0] != indexes[name], similarity_scores))

def vectorization(data, path, kernels):
    tfidfv = TfidfVectorizer()
    tfidf_data = tfidfv.fit_transform(data)
    matrix = kernels(tfidf_data, tfidf_data)
    if kernels is cosine_similarity:
        np.save(path, matrix)
    return matrix

def make_recomendations_with_genre(cbf_matrix, name = ""
                                   , count_recomendations = 10
                                   , kernels= cosine_similarity):
    if os.path.exists(CBF_GENRE_DATA) and kernels is cosine_similarity:
        similarity_matrix = np.load(CBF_GENRE_DATA)
    else:
        genres = cbf_matrix['Genres'].str.split(', ').astype(str)
        similarity_matrix = vectorization(genres, CBF_GENRE_DATA, kernels)
    
    anime_indexes = pd.Series(cbf_matrix.index, index=cbf_matrix['Name'])
    if name == "":
        name = np.random.choice(anime_indexes.index)
    try:
        similarity_scores = get_scores(similarity_matrix, anime_indexes, name)

        ind = similarity_scores[0: count_recomendations]
        recomendations = []
        
        for _, index in enumerate(ind):
            title = cbf_matrix[['Name', 'Genres', 'sypnopsis']].iloc[index[0]].tolist()
            title.append(index[1])
            recomendations.append(title)

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
        similarity_matrix = vectorization(synopsis, CBF_SYPNOPSIS_DATA, kernels)

    anime_indexes = pd.Series(cbf_matrix.index, index=cbf_matrix['Name'])
    if name == "":
        name = np.random.choice(anime_indexes.index)
    try:
        similarity_scores = get_scores(similarity_matrix, anime_indexes, name)

        ind = similarity_scores[0: count_recomendations]
        recomendations = []

        for _, index in enumerate(ind):
            title = cbf_matrix[['Name', 'Genres', 'sypnopsis']].iloc[index[0]].tolist()
            title.append(index[1])
            recomendations.append(title)

        return name, recomendations
    except:
        return None
    
def make_recomendations_with_genres_and_sypnopsis(cbf_matrix, name = ""
                                                  , count_recomendations = 10
                                                  , kernels= cosine_similarity):
    if os.path.exists(CBF_SYPNOPSIS_DATA) and kernels is cosine_similarity:
        similarity_sypnonpsis_matrix = np.load(CBF_SYPNOPSIS_DATA)
    else:
        synopsis = cbf_matrix['sypnopsis'].str.strip(',.!?:"()').str.split(' ').astype(str)
        similarity_sypnonpsis_matrix = vectorization(synopsis
                                                     , CBF_SYPNOPSIS_DATA
                                                     , kernels)

    if os.path.exists(CBF_GENRE_DATA) and kernels is cosine_similarity:
        similarity_genres_matrix = np.load(CBF_GENRE_DATA)
    else:
        genres = cbf_matrix['Genres'].str.split(', ').astype(str)
        similarity_genres_matrix = vectorization(genres
                                                 , CBF_GENRE_DATA, kernels)
    anime_indexes = pd.Series(cbf_matrix.index, index=cbf_matrix['Name'])
    if name == "":
        name = np.random.choice(anime_indexes.index)
    try:
        similarity_genres_scores = get_scores(similarity_genres_matrix
                                                , anime_indexes, name, False)
        similarity_sypnopsis_scores = get_scores(similarity_sypnonpsis_matrix
                                                    , anime_indexes, name, False)
        similarity_scores = [[x[0], x[1] * y[1]] for x,y in zip(similarity_genres_scores,similarity_sypnopsis_scores)]
        similarity_scores = sorted(similarity_scores
                                , key= lambda x: x[1], reverse= True)
        ind = similarity_scores[0: count_recomendations]
        recomendations = []

        for _, index in enumerate(ind):
            title = cbf_matrix[['Name', 'Genres', 'sypnopsis']].iloc[index[0]].tolist()
            title.append(index[1])
            recomendations.append(title)

        return name, recomendations
    except:
        return None