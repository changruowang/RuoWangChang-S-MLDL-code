#!/usr/bin/python
# -*- coding:utf-8 -*-
from io import StringIO
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import numpy as np
from myLR import myLineRegression
from myTool.myTools import DrawClassyPic
from mpl_toolkits.mplot3d import Axes3D

def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]

if __name__ == "__main__":
    # # 随机产生多分类数据，n_samples=样本数，n_features=x数据维度，centers=y分类数
    # x, y = datasets.make_blobs(n_samples=100, n_features=2, centers=4, cluster_std=1.0, center_box=(-10.0, 10.0),
    #                            shuffle=True, random_state=None)
    path = '..\\data\\8.iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    x, y = np.split(data, (4,), axis=1)
    x = x[:,(0,3)]

    # 调用数据预处理函数
    sm = myLineRegression.MySoftMax(learning_rate=0.05,n_iterations=3000)

    sm.fit(x,y)
    print(sm.theta)

    DrawClassyPic(x, y, sm.predict, x_label=u'花瓣长度', y_label=u'花瓣宽度', title=u'鸢尾花')


    # 这是一个梯度下降实现线性回归的例子

    names =["mpg","cylinders","displacement","horsepower",
            "weight","acceleration","model year","origin","car name"]

    path = '..\\data\\regressionData2.csv'

    cReader = pd.read_csv(path, delim_whitespace=True, names=names)

    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    ax.scatter(cReader["mpg"][:100],cReader["displacement"][:100],cReader["acceleration"][:100],color='y')
    ax.scatter(cReader["mpg"][100:250],cReader["displacement"][100:250],cReader["acceleration"][100:250],c='r')
    ax.scatter(cReader["mpg"][250:],cReader["displacement"][250:],cReader["acceleration"][250:],c='b')

    ax.set_zlabel('acceleration')  # 坐标轴
    ax.set_ylabel('displacement')
    ax.set_xlabel('mpg')
    plt.show()
    #
    # plt.scatter(cReader["mpg"], cReader["displacement"])
    # plt.xlabel('mpg')
    # plt.ylabel('displacement')
    # plt.show()
    #
    # trainX = cReader[['mpg', 'displacement']]
    # trainY = cReader['acceleration']
    #
    # X = np.mat(trainX)
    # Y = np.mat(trainY).T
    #
    # # 每一列的特征标准化(x - min) / (max - min)
    # for i in range(0, 2):
    #     X[:, i] = (X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))
    #
    # model = myLineRegression.MyLineRegression(learning_rate=0.005,n_iterations=500,UseGradient=True)
    # model.fit(X, Y)
    #
    # print("ti du xia jiang",model.Theta())
    #
    # x1 = np.linspace(X[:,0].min(),X[:,0].max(),100)
    # x2 = np.linspace(X[:,1].min(),X[:,1].max(),100)
    #
    #
    # x1,x2 = np.meshgrid(x1,x2)
    # finalTheta = model.Theta()
    # f = finalTheta[0,0] + finalTheta[1,0]*x1 + finalTheta[2,0]*x2
    #
    # fig = plt.figure()
    # Ax = Axes3D(fig)
    # Ax.plot_surface(x1, x2, f, rstride=1, cstride=1, cmap=cm.viridis,label='prediction')
    #
    # Ax.scatter(X[:100,0],X[:100,1],Y[:100,0],c='y')
    # Ax.scatter(X[100:250,0],X[100:250,1],Y[100:250,0],c='r')
    # Ax.scatter(X[250:,0],X[250:,1],Y[250:,0],c='b')
    #
    # Ax.set_zlabel('acceleration')  # 坐标轴
    # Ax.set_ylabel('displacement')
    # Ax.set_xlabel('mpg')
    #
    # plt.show()
    #
