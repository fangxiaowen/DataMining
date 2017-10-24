# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:13:55 2017
@author: Administrator
"""

"""
Features: uniform weighted word vectors
Dimension: 10 (PCA)
KNN: 5 neighbors, cosine (normalized eucilidean distance), distance (voted weighted by 1/distance)

Todo:
    Weight word vectors by TF-IDF
"""
#import re
import scipy
from numpy.linalg import norm
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import neighbors
from gensim.models import word2vec
import logging
import numpy as np
from sklearn.preprocessing import scale, normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import time
import string

from sklearn import cross_validation
import sys

# Get feature vector of a single review
def getWordVecsWithTf_Idf(i, train):   # i is the # of document
    vecs = []
    if train:
        meaningful_doc = meaningful_words_indoc[i]
        tfidf_m = tfidf_matrix
    else:
        meaningful_doc = meaningful_words_indoc_test[i]
        tfidf_m = tfidf_matrix_test
        
    tfidf_value = tfidf_m[i][0,tfidf_m[i].indices].toarray()[0]
    try:
        assert len(tfidf_value) == len(meaningful_doc)
    except AssertionError:
        print(len(tfidf_value), len(meaningful_doc))
        sys.exit(1)
        
    for j in range(len(meaningful_doc)):
    #for word in meaningful_words_indoc[i]:
        #word = word.replace('\n', '')
        try:
            vecs.append(model[meaningful_doc[j]] * tfidf_value[j])
        except KeyError:
            continue
    # vecs = np.concatenate(vecs)
    
    vecs = np.array(vecs, dtype = 'float')
    return vecs

"""
def getWordVecs(wordList):
    vecs = []
    for word in wordList:
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype = 'float')
"""

def buildVecs2(corpus, train):
    posInput = []
    for i in range(len(corpus)):
        resultList = getWordVecsWithTf_Idf(i, train)
        try:
            assert len(resultList) != 0
            resultArray = sum(np.array(resultList))/len(resultList)
            posInput.append(resultArray)
        except AssertionError:
            print(corpus[i], i)
            resultArray = np.array([0] * 100)
            posInput.append(resultArray)
            if train:
                label[i] = 0
            continue

    return posInput

"""     
def buildVecs(corpus, train):
    posInput = []
    i = -1
    for review in corpus:
        i += 1
        resultList = getWordVecs(review)
        if len(resultList) != 0:
            resultArray = sum(np.array(resultList))/len(resultList)
            posInput.append(resultArray)
        else:
            resultArray = np.array([0] * 100)
            print(i, review)
            if train:
                label[i] = 0
            posInput.append(resultArray)
    return posInput
"""
        
#load data
with open(r'C:\Users\Administrator\Desktop\1504108575_8218148_train_file.data', 'r') as f:
    train_data = f.readlines()
label = np.array([1 if i[0] == '+' else -1 for i in train_data])
train_text = [i[3:-1] for i in train_data]   
for i in range(len(train_text)):
    temp_text = train_text[i].lower()
    transtable = str.maketrans('', '', string.punctuation)
    train_text[i] = temp_text.translate(transtable)
train_wordlist = ["".join((char if char.isalpha() else " ") for char in s).split() for s in train_text]

# test data
with open(r'C:\Users\Administrator\Desktop\1504108575_8450022_test.data', 'r') as f:
    test_data = f.readlines()

test_text = [i[:-1] for i in test_data]   
for i in range(len(test_text)):
    temp_text = test_text[i].lower()
    transtable = str.maketrans('', '', string.punctuation)
    test_text[i] = temp_text.translate(transtable)
test_wordlist = ["".join((char if char.isalpha() else " ") for char in s).split() for s in test_text]

sys.exit(0)

# get TF-IDF matrix of corpus
obj = TfidfVectorizer()
tfidf_matrix = obj.fit_transform(train_text)
meaningful_words_indoc = obj.inverse_transform(tfidf_matrix)

# get TF-IDF matrix of corpus in test data

#obj = TfidfVectorizer()
tfidf_matrix_test = obj.transform(test_text)
#meaningful_words_indoc_test = obj.inverse_transform(tfidf_matrix_test)
meaningful_words_indoc_test = obj.inverse_transform(tfidf_matrix_test)

#feature extraction
model = word2vec.Word2Vec(train_wordlist, size = 100)

# get feature vector and Demension Reduction
posInput = buildVecs2(train_wordlist, True)
X = posInput[:]
X = np.array(X)
X = scale(X)



X_reduced = PCA(n_components = 10).fit_transform(X)
X_reduced = normalize(X_reduced)

# feature of test set
posInput_test = buildVecs2(test_wordlist, False)
X_test = posInput_test[:]
X_test = np.array(X_test)
X_test = scale(X_test)


X_test_reduced = PCA(n_components = 10).fit_transform(X_test)
X_test_reduced = normalize(X_test_reduced)


# knn classifier
reviews_tree = KDTree(X_reduced, leaf_size=2)    

start = time.clock()
#weights = np.array([1/2,1/4,1/8,1/16,1/16])

dist, ind = reviews_tree.query(X_test_reduced, k=6, metric=cosine_dis)
predict = np.sum((1 / (dist + 0.1)) * label[ind], axis=1)
end = time.clock()
print('time is : ', end - start)

final_result = ''
for i in predict:
    if i >= 0:
        final_result += '+1\n'
    else:
        final_result += '-1\n'
with open(r'C:\Users\Administrator\Desktop\knn_predict_amazon_first1000.data', 'w') as f:
    f.write(final_result)







# KNN in sklearn and CV in sklearn

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_reduced, label, test_size=0.4, random_state=0)
"""
clf = neighbors.KNeighborsClassifier(n_neighbors=7, weights='distance', metric=cosine_dis)
#clf.fit(X_train, y_train)
#clf.score(X_test, y_test)
ss = time.clock()
scores = cross_val_score(clf, X_reduced, label, cv=5)
ee = time.clock()
print(ee - ss)

clf.fit(X_reduced, label)
ss = time.clock()
pre = clf.predict(X_test_reduced)
ee = time.clock()
print(ee - ss)
"""

def cosine_dis(u, v):
    return 1.0 - np.dot(u, v) #/ (norm(u) * norm(v))
        

