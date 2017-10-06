# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 15:20:21 2017

@author: Administrator
"""

from sklearn.model_selection import cross_val_score
#from sklearn import neighbors
#from gensim.models import word2vec
#import logging
from sklearn.pipeline import Pipeline
import sys
import numpy as np
from sklearn.preprocessing import scale, normalize
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

#load data
with open(r'C:\Users\Administrator\Desktop\cs484_hw2\1505760800_9146957_train_drugs.data', 'r') as f:
    train_data = f.readlines()
label = np.array([1 if i[0] == '1' else 0 for i in train_data])
train_feature = [list(map(int, i[2:-1].split())) for i in train_data]
#train_wordlist = ["".join((char if char.isalpha() else " ") for char in s).split() for s in train_text]
with open(r'C:\Users\Administrator\Desktop\cs484_hw2\1505760800_9191542_test.data', 'r') as f:
    test_data = f.readlines()
test_feature = [list(map(int, i[:-1].split())) for i in test_data]


#construct sparse matrix
indptr, indices, data = np.array([0]), np.array([], dtype=int), np.array([], dtype=int)

for i in train_feature:
    indptr = np.append(indptr, indptr[-1] + len(i))
    indices = np.append(indices, i)
    data = np.append(data, np.ones(len(i)))
    
X_train = csr_matrix((data, indices, indptr), shape=(800, 100000))

indptr, indices, data = np.array([0]), np.array([], dtype=int), np.array([], dtype=int)

for i in test_feature:
    indptr = np.append(indptr, indptr[-1] + len(i))
    indices = np.append(indices, i)
    data = np.append(data, np.ones(len(i)))
    
X_test = csr_matrix((data, indices, indptr), shape=(350, 100000))

sys.exit(0)

# Classification with Bagging! and Boosting!
print('Okay we get here!')
w = 10
#clf = svm.SVC(class_weight='balanced', gamma=1/2000, cache_size=1000)
lsvc = svm.LinearSVC(C=1, penalty="l1", class_weight='balanced', dual=False).fit(X_train, label)
model = SelectFromModel(lsvc, prefit=True)

clf = Pipeline([
  ('feature_selection', model),
  ('classification', svm.Linear(C=0.2105, class_weight='balanced', gamma=1/5000, cache_size=1000))
])
    
s = time.clock()
bagging = BaggingClassifier(clf, max_samples=1.0, max_features=1.0)

#bdt = AdaBoostClassifier(clf, algorithm="SAMME", n_estimators=50)

#clf.fit(X_train, label)  
#pre = clf.predict(X_test)
bagging.fit(X_train, label)

#bdt.fit(X_train, label)

pre = bagging.predict(X_test)
#pre = bdt.fit(X_test)
e = time.clock()
# Cross Validation
#scores = cross_val_score(clf, X_train, label, cv=3, scoring='f1')
print('# of positive is :', pre[pre==1].shape[0])
print('weight is : ', w)
print('We finished! time is :', e - s)

# write it to csv file
with open(r'C:\Users\Administrator\Desktop\cs484_hw2\predicted_class.csv', 'w') as f:
    for i in pre:
        f.write(str(i) + '\n')
    










"""
# Demension Reduction
svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
svd.fit(X_train)
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(svd.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

X_train_reduced = svd.transform(X_train)

svd.fit(X_test)
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(svd.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

X_test_reduced = svd.transform(X_test)
""" 
    
    