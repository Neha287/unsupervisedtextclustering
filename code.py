# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 18:24:43 2018

@author: Neha
"""
from __future__ import print_function

import numpy as np
import pandas as pd
import nltk

#nltk.download()
from bs4 import BeautifulSoup
import re
import os
import codecs
from sklearn import feature_extraction
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import matplotlib.cm as cm




from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

print(__doc__)



import json

with open('C:\\Fall 2018\\AI Project\\Code\\arxivdataset\\arxivData.txt') as json_file:  
    data = json.load(json_file)
    
authors = []
for author in data:
    authors.append(author['author'])
authors10 = []
authors10 = authors[500:1000]

    
titles = []
for title in data:
    titles.append(title['title'])
titles10 = []
titles10 = titles[500:1000]

summaries = []
for summary in data:
    summaries.append(summary['summary'])
summaries10 = []
summaries10 = summaries[500:1000]

print(str(len(titles10)) + ' titles')
print(str(len(authors10)) + ' links')
print(str(len(summaries10)) + 'summaries')
#print(titles10)

#summaryrows = []
#summaryrows.append(summaries[:10])
#print(summaryrows)
# generates index for each item in the corpora (in this case it's just rank) and I'll use this for scoring later
ranks = []

for i in range(0,len(summaries10)):
    ranks.append(i)
    
# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')
# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in summaries10:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
    
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print(vocab_frame)

tfidf_vectorizer = TfidfVectorizer(max_df=1, max_features=200000,
                                 min_df=0, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(summaries10)
print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()

dist = 1 - cosine_similarity(tfidf_matrix)
distorsions = []
for k in range(3,7):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(tfidf_matrix)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(3,7), distorsions)
plt.grid(True)
plt.title('Elbow curve')

num_clusters = 5
kmeans= KMeans(n_clusters=num_clusters)

kmeans.fit(tfidf_matrix)

clusters = kmeans.labels_.tolist()
joblib.dump(kmeans,  'doc_cluster.pkl')
km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()
print(clusters)
papers = { 'title': titles10, 'author': authors10,'rank':ranks, 'summary': summaries10, 'cluster': clusters}

frame = pd.DataFrame(papers, index = [clusters] , columns = ['rank','author', 'title', 'cluster'])
frame['cluster'].value_counts()
#print(frame)
grouped = frame['rank'].groupby(frame['cluster'])

grouped.mean()
print("Top terms per cluster:")
print()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    for ind in order_centroids[i, :6]:
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()
    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print()
    print()
    import os  # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

MDS()

# two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
#strip any proper nouns (NNP) or plural proper nouns (NNPS) from a text
from nltk.tag import pos_tag

def strip_proppers_POS(text):
    tagged = pos_tag(text.split()) #use NLTK's part of speech tagger
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns
#set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: '#66a61e',6:'#e7298a'}
#set up cluster names using a dict
cluster_names = {0: 'C1', 
                 1: 'C2', 
                 2: 'C3', 
                 3: 'C4', 
                 4: 'C5',
                 5: 'C6',
                 6: 'C7'
                 
                  }
#%matplotlib inline
#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles10)) 

#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')
    ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

plt.show() #show the plot

silhouette_avg = silhouette_score(tfidf_matrix, clusters)
score = metrics.silhouette_score(tfidf_matrix, clusters, metric='euclidean')
test = []
test =   tfidf_matrix.toarray()
ch_score= metrics.calinski_harabaz_score(test,clusters )
print(ch_score)
print("For n_clusters =", num_clusters,
         "The average silhouette_score is :", silhouette_avg)
