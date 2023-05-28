import pickle
import os
from typing import Optional

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

ANIME_DIR = "anime_data/"

def make_recomendations_with_cf(cf_matrix, name = "",
                                count_recomendations = 10,
                                model_file = 'cf_model.sav'):
    if os.path.exists(ANIME_DIR + model_file):
        model = pickle.load(open(ANIME_DIR + model_file, 'rb'))
    else:
        cf_matrix_csr = csr_matrix(cf_matrix.values)
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(cf_matrix_csr)

        pickle.dump(model, open(ANIME_DIR + model_file, 'wb'))

    
    if name == "":
        name = np.random.choice(cf_matrix.index.values)
    try:
        index = cf_matrix.index.get_loc(name)
        distances, indexes = model.kneighbors(cf_matrix.
                                              iloc[index, :].
                                              values.reshape(1, -1),
                                              n_neighbors= count_recomendations)
        recomendations = []
        for i, ind in enumerate(indexes.flatten()):
            recomendations.append([cf_matrix.index[ind], distances.flatten()[i]])
        
        return name, recomendations
    except:
        return None