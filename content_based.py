# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 23:53:13 2019

@author: Avas Ansuman
"""

#recommendation based on similarity of content
#we will be using imdb dataset
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import re
df = pd.read_csv('https://query.data.world/s/uikepcpffyo2nhig52xxeevdialfl7')
df = df[['Title','Genre','Director','Actors','Plot']]
indices = pd.Series(df['Title'])
df['Key_words'] = ""
for index, row in df.iterrows():
    plot = row['Plot']
    stop_words = set(stopwords.words('english'))
    plot = re.sub(r'[^\w\s]','',plot)
    word_tokens = word_tokenize(plot) 
    row['Key_words']= [w for w in word_tokens if not w in stop_words]
    
df.drop(columns = ['Plot'], inplace = True)
#creating list of actors,genres,directors
df['Actors'] = df['Actors'].map(lambda x: x.split(',')[:3])
df['Genre'] = df['Genre'].map(lambda x: x.lower().split(','))
df['Director'] = df['Director'].map(lambda x: x.split(' '))

#merging first and second names of actors
for index, row in df.iterrows():
    row['Actors'] = [x.lower().replace(' ','') for x in row['Actors']]
#doing same for director names
for index, row in df.iterrows():
    row['Director'] = ''.join(row['Director']).lower()    
#creating a bag of words coloumn consisting of actors,director,plot,title and genre
df['bag_of_words'] = ''
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
        if (col != 'Director') & (col!= 'Title') :
            words = words + ' '.join(row[col])+ ' '
        else:
            words = words + row[col]+ ' '
    row['bag_of_words'] = words
    
df.drop(columns = [col for col in df.columns if col!= 'bag_of_words'], inplace = True)   

#after creating bag of words we use countvectorizer method to fnd the vector of bag of words
count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])

cosine_sim = cosine_similarity(count_matrix, count_matrix)
def recommend(title,cosine_sim=cosine_sim,df=df):
    recommendations=[]
    idx = indices[indices == title].index[0]  
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_10_indexes = list(score_series.iloc[1:11].index)
    for i in top_10_indexes:
        recommendations.append(indices[i])
    return recommendations

#write the movie name to find its recommendation
recommend('Inception')
    
    
