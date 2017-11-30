# encoding=utf-8

import pandas as pd
import numpy as np
import time

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cross_validation import train_test_split


if __name__ == '__main__':

    print("Start read data...")

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train.csv', header=0)  # 读取csv数据
    data = raw_data.values
    
    features = data[::, 1::]
    labels = data[::, 0]

    # 随机选取33%数据作为测试集，剩余为训练集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=0)

    time_2 = time.time()
    print('read data cost %f seconds' % (time_2 - time_1))

    print('Start training...')
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(train_features, train_labels) 
    time_3 = time.time()
    print('training cost %f seconds...' % (time_3 - time_2))

    print('Start predicting...')
    test_predict = neigh.predict(test_features)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))
    
    score = neigh.score(test_features, test_labels)
    print("The accruacy score is %f" % score)
