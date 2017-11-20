# encoding=utf-8

import pandas as pd
import random
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


class Perceptron(object):

    def __init__(self):
        self.learning_step = 0.001  # 学习率
        self.max_iteration = 5000  # 分类正确上界，当分类正确的次数超过上界时，认为已训练好，退出训练

    def train(self, features, labels):        
       
        # 初始化w,b为0,b在最后一位
        self.w = [0.0] * (len(features[0]) + 1)

        correct_count = 0  # 分类正确的次数

        while correct_count < self.max_iteration:

            # 随机选取数据(xi,yi)
            index = random.randint(0, len(labels) - 1) 
            x = list(features[index])
            x.append(1.0)  # 加上1是为了与b相乘
            y = 2 * labels[index] - 1  # label为1转化为正实例点+1，label为0转化为负实例点-1

            # 计算w*xi+b
            wx = sum([self.w[j] * x[j] for j in range(len(self.w))])

            # 如果yi(w*xi+b) > 0 则分类正确的次数加1
            if wx * y > 0:
                correct_count += 1
                continue

            # 如果yi(w*xi+b) <= 0 则更新w(最后一位实际上b)的值
            for i in range(len(self.w)):
                self.w[i] += self.learning_step * (y * x[i])

    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        return int(wx > 0)  # w*xi+b>0则返回返回1,否则返回0
    
    def predict(self, features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels


if __name__ == '__main__':

    print("Start read data")

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train_binary.csv', header=0)  # 读取csv数据，并将第一行视为表头，返回DataFrame类型
    data = raw_data.values
    
    features = data[::, 1::]
    labels = data[::, 0]

    # 避免过拟合，采用交叉验证，随机选取33%数据作为测试集，剩余为训练集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=0)

    time_2 = time.time()
    print('read data cost %f seconds' % (time_2 - time_1))

    print('Start training')
    p = Perceptron()
    p.train(train_features, train_labels)

    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))

    print('Start predicting')
    test_predict = p.predict(test_features)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))

    score = accuracy_score(test_labels, test_predict)
    print("The accruacy score is %f" % score)
