# lihang_algorithms
用python3实现李航老师的《统计学习方法》中所提到的算法
<br><br>实验数据：MNIST数据集
<br>官方下载地址：http://yann.lecun.com/exdb/mnist/
<br>kaggle中处理好的数据：https://www.kaggle.com/c/digit-recognizer/data

## 第二章 感知机
适用问题：二类分类
<br>代码：[perceptron/perceptron.py](https://github.com/fuqiuai/lihang_algorithms/blob/master/perceptron/perceptron.py)
<br>运行结果：
<br>![](https://raw.githubusercontent.com/fuqiuai/lihang_algorithms/master/imgs/perceptron_result.png)

## 第三章 k邻近法
适用问题：多类分类
<br>三个基本要素：k值的选择、距离度量及分类决策规则
<br>代码：[knn/knn.py](https://github.com/fuqiuai/lihang_algorithms/blob/master/knn/knn.py)
<br>运行结果：
<br>![](https://raw.githubusercontent.com/fuqiuai/lihang_algorithms/master/imgs/knn_result.png)

## 第四章 朴素贝叶斯法
适用问题：多类分类
<br>基于贝叶斯定理和特征条件独立假设
<br>代码：[naive_bayes/naive_bayes.py](https://github.com/fuqiuai/lihang_algorithms/blob/master/naive_bayes/naive_bayes.py)
<br>运行结果：
<br>![](https://raw.githubusercontent.com/fuqiuai/lihang_algorithms/master/imgs/naive_bayes_result.png)