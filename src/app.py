# imports:
import os
import ast
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# load datasets:
#movies_url = 'https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv'
#credits_url = 'https://github.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/blob/main/tmdb_5000_credits.zip'
movies_raw = pd.read_csv('../data/raw/movies.csv')
credits_raw = pd.read_csv('../data/raw/credits.csv')

movies = movies_raw.copy()
credits = credits_raw.copy()

# Merge both datasets on 'Title'
movies = movies.merge(credits, on='title')

# Keep only some columns:
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# As there are only 3 missing values in the 'overview' column, will drop them.
movies.dropna(inplace = True)

#####################################################################################################
# Will Convert these columns using a function to obtain only the genres and without a json format.
# We are only interested in the values of the 'name' keys:
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count < 3:
            L.append(i['name'])
        count +=1  
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# Convert 'genres' & 'keyword' jsons to lists
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Convert 'cast' json to lists with a max of 3 elements:
movies['cast'] = movies['cast'].apply(convert3)

# From 'crew' we want the 'name' if 'job' is 'Director':
movies['crew'] = movies['crew'].apply(fetch_director)

# Convert 'overview' to list using split():
movies['overview'] = movies['overview'].apply(lambda x : x.split())
#####################################################################################################
#####################################################################################################
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

# Apply to 'genres', 'cast', 'crew', 'keywords':
for column in ['genres', 'cast', 'crew', 'keywords'] :
    movies[column] = movies[column].apply(collapse)
#####################################################################################################

# Will combine the previous converted columns into only one single column: 'tags'
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_movies = movies[['movie_id','title','tags']]

# Delete the commas that separate the words:
new_movies['tags'] = new_movies['tags'].apply(lambda x :" ".join(x))

new_movies.to_csv('../data/processed/movies.csv')


# Vectorization:
model_KNN = CountVectorizer(max_features=5000 , stop_words='english')
vectors = model_KNN.fit_transform(new_movies['tags']).toarray()

# distance: cosine similarity (word vectors)
similarity = cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])),reverse =True , key = lambda x:x[1])[1:6]

# export the model:
# We save the model with joblib
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../models/KNN_movies.pkl')

joblib.dump(model_KNN, filename)

#####################################################################################################
# Implement Recommender
def recommend(movie) :
    nunique_movies = new_movies['title'].unique()
    if movie in nunique_movies :
        #fetching the movie index:
        movie_index = new_movies[new_movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)),reverse =True , key=lambda x:x[1])[1:6]
        print(f'List of similar movies for {movie}:')
        for i in movie_list:
            print(new_movies.iloc[i[0]].title)
    else :
        print(f'The movie "{movie}" was not found on the database. Sorry :)')

# recommend('choose a movie here')
movie = input('Select a Movie: ')
recommend(movie)
#####################################################################################################
