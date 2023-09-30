# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 20:05:58 2023

@author: Aditi
"""

import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

movies = pd.read_csv("C:/Users/Aditi/Downloads/archive(19)/tmdb_5000_movies.csv")

creditss = pd.read_csv("C:/Users/Aditi/Downloads/archive(19)/tmdb_5000_credits.csv")

movies.head()

creditss.head()

movies = movies.merge(creditss, on='title')

movies.head()

#genres, #id, #keywords, #title, #overview, #cast, #crew

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

movies.info()
movies.head()

movies.isnull().sum()
movies.dropna(inplace=True)
movies.isnull().sum()
movies.duplicated().sum()
movies.iloc[0].genres

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies.head()

movies['keywords'] = movies['keywords'].apply(convert)
movies.head()

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if i != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
        
    return L

movies['cast'] = movies['cast'].apply(convert3)
movies.head()

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)
movies.head()

print(movies['overview'][0])

movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies.head()

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

print(movies.head())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

print(movies.head())

new_df = movies[['movie_id', 'title', 'tags']]

print(new_df)

new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

print(new_df['tags'][0])

new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

print(new_df.head())

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

print(vectors)

ps = PorterStemmer()

def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

print(similarity)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        
print(recommend('Avatar'))

pickle.dump(new_df, open('movies.pkl','wb'))

new_df['title'].values

pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))

pickle.dump(similarity,open('similarity.pkl', 'wb'))