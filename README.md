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

## 第五章 决策树
适用问题：多类分类
<br>三个步骤：特征选择、决策树的生成和决策树的剪枝
<br>常见的决策树算法有：ID3、C4.5和CART
<br><br>ID3算法代码：[decision_tree/ID3.py](https://github.com/fuqiuai/lihang_algorithms/blob/master/decision_tree/ID3.py)
<br>运行结果：
<br>![](https://raw.githubusercontent.com/fuqiuai/lihang_algorithms/master/imgs/ID3_result.png)
<br><br>C4.5算法代码：[decision_tree/C45.py](https://github.com/fuqiuai/lihang_algorithms/blob/master/decision_tree/C45.py)
<br>运行结果：
<br>![](https://raw.githubusercontent.com/fuqiuai/lihang_algorithms/master/imgs/C45_result.png)

## 第六章 逻辑斯谛回归
适用问题：多类分类
<br>与感知器算法贼像：
<br>感知器算法
> 当 Y = 1 时，wT⋅x 尽量等于 +1  
> 当 Y = 0 时， wT⋅x 尽量等于 -1

<br>而逻辑斯谛算法
> 当 Y = 1 时，wT⋅x 尽量等于 +∞  
> 当 Y = 0 时， wT⋅x 尽量等于 −∞

<br>代码(此处只实现二项逻辑斯谛回归)：[logistic_regression/logistic_regression.py](https://github.com/fuqiuai/lihang_algorithms/blob/master/logistic_regression/logistic_regression.py)
<br>运行结果：
<br>![](https://raw.githubusercontent.com/fuqiuai/lihang_algorithms/master/imgs/logistic_regression_result.png)

## 第六章 最大熵模型
适用问题：多类分类
$$ f(x,y)=\left\{\begin{matrix} 1 & (x,y)\in train set \\ 0 & else \end{matrix}\right. $$
<br>代码：[maxEnt/maxEnt.py](https://github.com/fuqiuai/lihang_algorithms/blob/master/maxEnt/maxEnt.py)
<br>运行结果：
<br>![](https://raw.githubusercontent.com/fuqiuai/lihang_algorithms/master/imgs/maxEnt_result.png)