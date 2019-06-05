#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import log
import numpy as np
from sklearn.model_selection import train_test_split
from myDT.myDecisionTree import createTree, classifyAll
from myTool.myTools import createPlot,switch_container

def createTestSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    testSet = [['晴', 35.4, 'high', 'false'],
               ['晴', 4.7, 'normal', 'false'],
               ['雨', 23.3, 'normal', 'false'],
               ['晴', 25.7, 'normal', 'true'],
               ['云', 22.7, 'high', 'true'],
               ['云', 26.4, 'normal', 'false'],
               ['雨', 23.3, 'high', 'true']]

    return testSet


def createDataSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    dataSet = [['晴', 35.6, 'high', 'false', 'N'],
               ['晴', 31.1, 'high', 'true', 'N'],
               ['云', 30.5, 'high', 'false', 'Y'],
               ['雨', 25.8, 'high', 'false', 'Y'],
               ['雨', 20.4, 'normal', 'false', 'Y'],
               ['雨', 3.3, 'normal', 'true', 'N'],
               ['云', 23.3, 'normal', 'true', 'Y']
               ]
    labels = [b'outlook', b'temperature', b'humidity', b'windy']
    return dataSet, labels

def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]


if __name__ == '__main__':
    # print(type(iris_type(b'Iris-setosa')))
    path = '..\\data\\8.iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=1)

    data_testX, data_testY = np.split(data_test, (4,), axis=1)

    labels = ['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']
    # dataSet, labels = createDataSet()
    # testDataSet = createTestSet()
    labelsTmp = labels[:]

    decisionTree = createTree(data_train.tolist(), labelsTmp)
    createPlot(decisionTree)
    y_hat = np.array(classifyAll(decisionTree, labels, data_testX.tolist()))
    result = (y_hat == data_testY.T)
    acc = np.mean(result)
    print(('准确度: %.2f%%' % (100 * acc)))
    print(switch_container(decisionTree))
    createPlot(decisionTree)
    # print('classifyResult:/n', )
