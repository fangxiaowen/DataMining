# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 22:48:41 2017

@author: Administrator
"""

import numpy as np
from surprise import SVD, SVDpp, Dataset, Reader, evaluate, print_perf
import time


# algo = SVDpp()  # too slow
algo2 = SVD()

reader = Reader(line_format='user item rating', sep='\x20')
data = Dataset.load_from_file(r'C:\Users\Administrator\Desktop\cs484_hw2\HW4\train.data', reader=reader)
data.split(n_folds=5)

data_test = np.loadtxt(r'C:\Users\Administrator\Desktop\cs484_hw2\HW4\test.data')

trainset = data.build_full_trainset()

# algo.train(trainset)   # too slow    12:40
s = time.clock()
algo2.train(trainset)
e = time.clock()
print(e - s)

pred = []
for i in data_test:
    pred.append(algo2.predict(str(int(i[0])), str(int(i[1])))[3])