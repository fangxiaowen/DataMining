# -*- coding: utf-8 -*-
from sklearn.model_selection import cross_val_score
#from sklearn import neighbors
#from gensim.models import word2vec
#import logging
import pickle
from sklearn.pipeline import Pipeline
import sys
import numpy as np
#from sklearn.preprocessing import scale, normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from sklearn.neighbors import KDTree
import time
from sklearn import cross_validation
from scipy.sparse import csr_matrix
from sklearn import svm
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, normalize
from sklearn.metrics import pairwise_distances
import scipy.cluster
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import Kmeans     # Kmeans class I write


import matplotlib.pyplot as plt


# preprocessing input to get sparse matrix
with open(r'C:\Users\Administrator\Desktop\cs484_hw2\HW3\input.data', 'r') as f:
    X_raw = f.readlines()

data, row_ind, col_ind = [], [], []
for i, line in enumerate(X_raw):
    li = line.strip().split()
    #print('length is ', len(li)//2)
    row_tmp = [i] * (len(li)//2)
    col_tmp = [li[j] for j in range(len(li)) if (j % 2 == 0)]
    data_tmp = [li[j] for j in range(len(li)) if (j % 2 == 1)]
    #row_ind.append(row_tmp)
    row_ind.extend(row_tmp)
    #col_ind.append(col_tmp)
    col_ind.extend(col_tmp)
    #data.append(data_tmp)
    data.extend(data_tmp)

data = list(map(int, data))
row_ind = list(map(int, row_ind))
col_ind = list(map(int, col_ind))

X = csr_matrix((data, (row_ind, col_ind)))
del X_raw, data, row_ind, col_ind


# feature extraction
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X)
tfidf_norm = normalize(tfidf)
#with open(r'C:\Users\Administrator\Desktop\cs484_hw2\HW3\train_tfidf_sparse.data', 'wb') as f:
#    pickle.dump(tfidf,f)


svd = TruncatedSVD(4000)
lsa = make_pipeline(svd, Normalizer(copy=False))
X_lsa = lsa.fit_transform(tfidf)
#with open(r'C:\Users\Administrator\Desktop\cs484_hw2\HW3\train__reduced.data', 'wb') as f:
#    pickle.dump(X_lsa,f)

explained_variance = svd.explained_variance_ratio_.sum()
print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))

sys.exit(0)


# Do KMeans clustering
#kmeans = KMeans(n_clusters=7, random_state=0).fit(X_lsa)

#kmeans_2 = KMeans(n_clusters=7).fit(tfidf)

kmeans_mine = Kmeans.KMeans(n_clusters=7, n_init=15, init='k-means++').fit(X_lsa)

#km = MiniBatchKMeans(n_clusters=7, init='k-means++', n_init=1, init_size=1000, batch_size=1000)

#db = DBSCAN(eps=1.3, min_samples=100).fit(X)

#knn_graph = kneighbors_graph(X_lsa, 100, include_self=False)

#clustering = AgglomerativeClustering(linkage='ward', n_clusters=7, memory=r'E:\cache')
#clustering.fit(X_lsa)
with open(r'C:\Users\Administrator\Desktop\cs484_hw2\HW3\text_clu.txt', 'w') as f:
    for i in kmeans_mine:
        f.write(str(int(i)) + '\n')    
        
# Plot





