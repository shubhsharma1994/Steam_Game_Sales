#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:27:20 2019

@author: shubhamsharma
"""

import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import re
from datetime import date 
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import heapq
import pandas as pd
import html
import numpy as np


import nltk
 
df_main = pd.read_csv("./steam.csv")
df_req = pd.read_csv("./steam_requirements_data.csv")
df_media = pd.read_csv("./steam_media_data.csv")
df_desc = pd.read_csv("./steam_description_data.csv")


df_main['Action']= 0    
for i in range(len(df_main['appid'])):
    y = df_main['genres'][i].split(',')
    
    if 'Action' in y:
        df_main['Action'][i] = 1


df_main.to_csv("main_act.csv")

df_2 = pd.DataFrame( columns = ['appid']) 
  
df_2["appid"]=df_main[df_main['english'] == 1]['appid']

result_desc = pd.merge(df_2,df_main[['appid', 'genres']],
                 on='appid')


result_desc['Action']= 0    
for i in range(len(result_desc['appid'])):
    y = result_desc['genres'][i].split(',')
    
    if 'Action' in y:
        result_desc['Action'][i] = 1
result_desc.to_csv("main_act.csv")



lines = []

for i in range(len(result_desc)):
    lines.append(result_desc["detailed_description"][i])
    
import re
def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

line_3=[]
for j in range(len(lines)):
    line_3.append(str.strip(striphtml(lines[j])).lower())

    
#TOKENIZE

token = []

for j in range(len(line_3)):
    token.append(nltk.word_tokenize(line_3[j]))

#remove stop words
    
from nltk.corpus import stopwords

stop_words_removed=[]

for m in range(len(token)):
    stop_words_removed.append([t1 for t1 in token[m] if not t1 in stopwords.words('english') if t1.isalpha()])
 
    
print(stop_words_removed)        
                 
#lemmatization 

#DO LEMMATIZATION FIRST AND THEN DO STEMMING - WORKS BETTER


lemmatizer = nltk.stem.WordNetLemmatizer()
lemm=[]

for k in range(len(stop_words_removed)):
    lemm.append([lemmatizer.lemmatize(t) for t in stop_words_removed[k] if t.isalpha()])
             
    
# reviews in Count vectors

from sklearn.feature_extraction.text import CountVectorizer

stop=[]
for j in range(len(lemm)):
    s=" ".join(stop_words_removed[j])
    stop.append(s)

vectorizer2 = CountVectorizer(ngram_range=(1,2),min_df =5,max_df=1000)
vectorizer2.fit(stop)
print(vectorizer2.vocabulary_)
v1 = vectorizer2.transform(stop)
print(v1.toarray())







# reviews in TD-IDF vectors

from sklearn.feature_extraction.text import TfidfVectorizer

stop=[]
for j in range(len(stop_words_removed)):
    s=" ".join(stop_words_removed[j])
    stop.append(s)


vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df =5,max_df=1000)
vectorizer.fit(stop)
print(vectorizer.vocabulary_)
v = vectorizer.transform(stop)
print(v.toarray())


#df_vect=pd.DataFrame(v1.toarray())


# LDA Model

terms = vectorizer2.get_feature_names()
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=6, max_iter=20, learning_method='online', learning_offset=50.,random_state=0).fit(v1)
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx))
    print(" ".join([terms[i] for i in topic.argsort()[:-4-1:-1]]))
    

# LDA Model
topics=[]
for i in range(len(df_2)):
    topics.append(lda.transform(v1[i])[0])
 
topic_dist = lda.transform(v1)

df_topic=pd.DataFrame(topic_dist)

df_topic.to_csv('Topics.csv', header=True, index=False) 



