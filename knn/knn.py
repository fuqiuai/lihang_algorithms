# encoding=utf-8

import pandas as pd
import numpy as np
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

def Predict(testset, trainset, train_labels):
    predict = []  # 保存测试集预测到的label，并返回
    count = 0  # 当前测试数据为第count个

    for test_vec in testset:
        # 输出当前运行的测试用例坐标，用于测试
        count += 1
        print("the number of %d is predicting..."%count)

        knn_list = []       # 当前k个最近邻居
        max_index = -1      # 当前k个最近邻居中距离最远点的坐标
        max_dist = 0        # 当前k个最近邻居中距离最远点的距离

        # 初始化knn_list,将前k个点的距离放入knn_list中
        for i in range(k):
            label = train_labels[i]
            train_vec = trainset[i]
            dist = np.linalg.norm(train_vec - test_vec)         # 计算两个点的欧氏距离
            knn_list.append((dist, label))

        # 剩下的点
        for i in range(k, len(train_labels)):
            label = train_labels[i]
            train_vec = trainset[i]
            dist = np.linalg.norm(train_vec - test_vec)         # 计算两个点的欧氏距离

            # 寻找k个邻近点中距离最远的点
            if max_index < 0:
                for j in range(k):
                    if max_dist < knn_list[j][0]:
                        max_index = j
                        max_dist = knn_list[max_index][0]

            # 如果当前k个最近邻中存在点距离比当前点距离远，则替换
            if dist < max_dist:
                knn_list[max_index] = (dist, label)
                max_index = -1
                max_dist = 0

        
        # 统计选票
        class_total = k
        class_count = [0 for i in range(class_total)]
        for dist, label in knn_list:
            class_count[label] += 1

        # 找出最大选票
        mmax = max(class_count)

        # 找出最大选票标签
        for i in range(class_total):
            if mmax == class_count[i]:
                predict.append(i)
                break

    return np.array(predict)

k = 10  # 选取k值

if __name__ == '__main__':

    print("Start read data")

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train.csv', header=0)  # 读取csv数据
    data = raw_data.values
    
    features = data[::, 1::]
    labels = data[::, 0]

    # 避免过拟合，采用交叉验证，随机选取33%数据作为测试集，剩余为训练集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=0)

    time_2 = time.time()
    print('read data cost %f seconds' % (time_2 - time_1))

    print('Start training')
    print('knn need not train')

    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))

    print('Start predicting')
    test_predict = Predict(test_features, train_features, train_labels)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))

    score = accuracy_score(test_labels, test_predict)
    print("The accruacy score is %f" % score)
