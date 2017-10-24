# -*- coding: utf-8 -*-
import numpy as np
import Kmeans     # Kmeans class I write
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

X = np.loadtxt(r'C:\Users\Administrator\Desktop\cs484_hw2\HW3\1507225115_9547634_iris.data')

clf = Kmeans.KMeans(n_init=10, n_clusters=3, init='k-means')

clf.fit(X)

#cluster = AgglomerativeClustering(linkage='complete', n_clusters=3, memory=r'E:\cache')
#cluster.fit(X)

print(clf._labels)

with open(r'C:\Users\Administrator\Desktop\cs484_hw2\HW3\iris_clu.txt', 'w') as f:
    for i in clf._labels:
        f.write(str(int(i)) + '\n')