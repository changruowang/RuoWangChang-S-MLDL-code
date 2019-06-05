#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import log
import numpy as np
from sklearn.model_selection import train_test_split
# from mySVM.mySVM import SVM,RBF
from mySVM.mySVM_L import SVM,RBF
from myTool.myTools import DrawClassyPic

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return np.array(dataMat),np.array(labelMat)

X,Y = loadDataSet('testSetRBF.txt')
kernel = RBF(gamma=1)
model = SVM(C=100, tol=0.000001, kernel=kernel)
model.fit(X, Y)
y_hat = model.predict(X)
score = y_hat==Y
print('SVM分类正确率:'+str(np.mean(score)))
DrawClassyPic(X, Y, classy_fun=model.predict,title=u'SVM Guss核测试')



