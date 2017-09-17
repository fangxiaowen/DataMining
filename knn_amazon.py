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
from sklearn.model_selection import cross_val_score

from sklearn import neighbors
from gensim.models import word2vec
import logging
import numpy as np
from sklearn.preprocessing import scale, normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import time
from sklearn import cross_validation

def getWordVecs(wordList):
    vecs = []
    for word in wordList:
        #word = word.replace('\n', '')
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    # vecs = np.concatenate(vecs)
    return np.array(vecs, dtype = 'float')

def buildVecs(corpus):
    posInput = []
    for review in corpus:
        resultList = getWordVecs(review)
        if len(resultList) != 0:
            resultArray = sum(np.array(resultList))/len(resultList)
            posInput.append(resultArray)
        else:
            resultArray = np.array([0] * 100)
            posInput.append(resultArray)
    return posInput

        
#load data
with open(r'C:\Users\Administrator\Desktop\1504108575_8218148_train_file.data', 'r') as f:
    train_data = f.readlines()
label = np.array([1 if i[0] == '+' else -1 for i in train_data])
train_text = [i[3:-1] for i in train_data]   
train_wordlist = ["".join((char if char.isalpha() else " ") for char in s).split() for s in train_text]

with open(r'C:\Users\Administrator\Desktop\1504108575_8450022_test.data', 'r') as f:
    test_data = f.readlines()

test_text = [i[:-1] for i in train_data]   
test_wordlist = ["".join((char if char.isalpha() else " ") for char in s).split() for s in test_text]



#feature extraction
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = word2vec.Word2Vec(train_wordlist, size = 100)
# save model
model.save(r"C:\Users\Administrator\Desktop\corpus.model")
# load model
# model = word2vec.Word2Vec.load("corpus.model")

# 以一种C语言可以解析的形式存储词向量
#model.save_word2vec_format("corpus.model.bin", binary = True)
# 对应的加载方式
# model = word2vec.Word2Vec.load_word2vec_format("corpus.model.bin", binary=True)

# get feature vector and Demension Reduction
posInput = buildVecs(train_wordlist)
X = posInput[:]
X = np.array(X)
X = scale(X)

pca = PCA()
pca.fit(X)
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')

X_reduced = PCA(n_components = 10).fit_transform(X)
X_reduced = normalize(X_reduced)

# feature of test set
posInput_test = buildVecs(test_wordlist)
X_test = posInput_test[:]
X_test = np.array(X_test)
X_test = scale(X_test)

pca.fit(X_test)

X_test_reduced = PCA(n_components = 10).fit_transform(X_test)
X_test_reduced = normalize(X_test_reduced)


# knn classifier
reviews_tree = KDTree(X_reduced, leaf_size=2)    

start = time.clock()
#weights = np.array([1/2,1/4,1/8,1/16,1/16])

dist, ind = reviews_tree.query(X_test_reduced, k=5)
predict = np.sum((1 / (dist + 0.1)) * label[ind], axis=1)
end = time.clock()
print('time is : ', end - start)

final_result = ''
for i in predict:
    if i >= 0:
        final_result += '+1\n'
    else:
        final_result += '-1\n'
with open(r'C:\Users\Administrator\Desktop\knn_predict_amazon_4.data', 'w') as f:
    f.write(final_result)



# KNN in sklearn and CV in sklearn

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_reduced, label, test_size=0.4, random_state=0)

clf = neighbors.KNeighborsClassifier(n_neighbors=15, weights='distance')
#clf.fit(X_train, y_train)
#clf.score(X_test, y_test)

scores = cross_val_score(clf, X_reduced, label, cv=7)

