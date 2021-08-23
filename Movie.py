# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 18:49:54 2021

@author: APPU
"""

#import numpy as np 
import pandas as pd

from PIL import Image
import streamlit as st

import re
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel

st.write("""
### Movie Recommendation System
""")


image1=Image.open('logo.jpg')

st.image(image1,caption=' ',use_column_width=True)


st.write("""This is a content based Movie recommendation system
         based on a Kaggle Dataset""")

data=pd.read_csv("netflix_titles.csv")

st.subheader('Sample Training Data: ')
st.dataframe(data.head())


data.drop(["show_id","director","cast","country","date_added","release_year","rating","duration"],axis=1,inplace=True)
data.head(10)

data['listed_in'] = [re.sub(r'[^\w\s]', '', t) for t in data['listed_in']]
data['description'] = [re.sub(r'[^\w\s]', '', t) for t in data['description']]

data['listed_in'] = [t.lower() for t in data['listed_in']]
data['description'] = [t.lower() for t in data['description']]

data["combined"] = data['listed_in'] + '  ' + data['title'] + ' ' + data['description'] 
data.drop(["description","listed_in","type"],axis=1,inplace=True)
data.head()

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(data["combined"])
cosine_similarities = linear_kernel(matrix,matrix)
movie_title = data['title']
indices = pd.Series(data.index, index=data['title'])

def content_recommender(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return movie_title.iloc[movie_indices]



#st.write("""Enter a movie which you liked in the below box:""")
user_input = st.text_input("""Enter a movie which you liked!""", "Tarzan")
st.write("""The below are the recommended movies: """)
st.write(content_recommender(user_input).head(10))