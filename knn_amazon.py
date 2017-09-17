# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 10:13:55 2017

@author: Administrator
"""
#import re
from gensim.models import word2vec
import logging
import numpy as np
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import time

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
label = [1 if i[0] == '+' else 0 for i in train_data]
train_text = [i[3:-1] for i in train_data]   
train_wordlist = ["".join((char if char.isalpha() else " ") for char in s).split() for s in train_text]









#feature extraction

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = word2vec.Word2Vec(train_wordlist, size = 100)
# 保存模型，以便重用
model.save(r"C:\Users\Administrator\Desktop\corpus.model")
# 对应的加载方式
# model = word2vec.Word2Vec.load("corpus.model")

# 以一种C语言可以解析的形式存储词向量
#model.save_word2vec_format("corpus.model.bin", binary = True)
# 对应的加载方式
# model = word2vec.Word2Vec.load_word2vec_format("corpus.model.bin", binary=True)


posInput = buildVecs(train_wordlist)
negInput = buildVecs(train_wordlist)
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

X_reduced = PCA(n_components = 50).fit_transform(X)

#Learn it using knn
with open(r'C:\Users\Administrator\Desktop\1504108575_8450022_test.data', 'r') as f:
    test_data = f.readlines()

test_text = [i[:-1] for i in train_data]   
test_wordlist = ["".join((char if char.isalpha() else " ") for char in s).split() for s in test_text]
X_test = buildVecs(test_wordlist)
X_test = np.array(X_test)
X_test = scale(X_test)

pca.fit(X_test)

X_test_reduced = PCA(n_components = 50).fit_transform(X_test)


reviews_tree = KDTree(X_reduced, leaf_size=2)    
result = []

start = time.clock()
for sample in X_test_reduced:
    dist, ind = reviews_tree.query(sample.reshape(1,-1), k=5)
    #weight = 1/2
    neighbors = label[ind][0]
    predict = 1/2 * neighbors[0] + 1/4 * neighbors[1] + 1/8 * neighbors[2] + 1/16 * neighbors[3] + 1/16 * neighbors[4]
    result.append(predict)
end = time.clock()
print('time is : ', end - start)

final_result = ''
for i in result:
    if i >= 0.5:
        final_result += '+1\n'
    else:
        final_result += '-1\n'
with open(r'C:\Users\Administrator\Desktop\knn_predict_amazon.data', 'w') as f:
    f.write(final_result)