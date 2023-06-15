import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

ANIME_DIR = "app/anime_data/"

CBF_GENRE_DATA = 'app/data/cbf_genre.npy'

CBF_SYPNOPSIS_DATA = 'app/data/cbf_sypnopsis.npy'

CBF_CLUSTER_MODEL = 'app/models/kmeans.sav'

OPTIMUM_NUM_CLUSTER = 20

def preprocessing():
    sypnopsis_data = pd.read_csv(ANIME_DIR + 'anime_with_synopsis.csv')
    dict = {'Unknown' : 0}
    sypnopsis_data['Score'] = sypnopsis_data['Score'] \
                            .apply(lambda x : dict[x] if x == 'Unknown' else x)\
                            .astype(float)

    vectorize_data = TfidfVectorizer().fit_transform(sypnopsis_data['Genres']
                                                        .str.split(', ')
                                                        .astype(str))
    if os.path.exists(CBF_CLUSTER_MODEL):
        model = pickle.load(open(CBF_CLUSTER_MODEL, 'rb'))
    else:
        
        model = KMeans(n_clusters= OPTIMUM_NUM_CLUSTER, random_state= 69
                                                    , n_init= 15)
        model.fit(vectorize_data)

    res_model = model.predict(vectorize_data)

    sypnopsis_data.loc[:, 'cluster'] = res_model

    images_links = pd.read_csv('app/static/images_links.csv', index_col='id')
    
    return sypnopsis_data, images_links