"# DataMining" These are projects from CS484--Data Mining course. Created by
Xiaowen Fang

Amazon_Review:

This project does a polarized sentiment analysis on product reviews on amazon. I
use word2vec to transfer text reviews to word vectors. And use PCA to do
dimension reduction. Then use KNN on testing set.

Medical_Molecule:

This project predicts weather a created molecule works for a certain kind of
drug. Each sample is a sparse (100,000 features) feature vector representing a
molecule. 800 samples in total. I use SVM for this one.

News_clustering:

This project aims to cluster news to different topics. I preprocess the data to a tf-idf matrix. Then use hierarchical
clustering with ward measurement metric.

Rating_movies:

This project predicts the ratings of each movie rated by users. I use matrix
factorization (SVD).  This is about recommender system. Dataset is from
MovieLens.

image_classification_kaggle_cdiscount:

Adapted from https://www.kaggle.com/ezietsman/inception-v3-finetune

This is for a kaggle competition : https://www.kaggle.com/c/cdiscount-image-
classification-challenge


Using inception v3 model (a CNN model) from keras. Using existed weights trained
on imageNet dataset. Only train the last several fully-connected layers on
cdiscount dataset.

