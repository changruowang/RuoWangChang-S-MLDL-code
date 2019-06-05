#!/usr/bin/python
# -*- coding:utf-8 -*-
from myKdTree.KdTree import kdTree
#from myTool.drawBinaryTrees import treeWriter
from myKMeans.myKMeans import MyKMeansPlus
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = [u'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    """
    k均值聚类测试
    """
    dateA, y = make_blobs(n_samples=1000, n_features=2, centers=3,
                                        cluster_std=[1.0,2.0,2.5], random_state=None)

    km = MyKMeansPlus(maxIter=100)
    y1 = km.fit(dateA, 3)

    plt.figure()
    sub1 = plt.subplot(121)
    sub1.scatter(dateA[:, 0], dateA[:, 1], c=y)
    sub2 = plt.subplot(122)
    sub2.scatter(dateA[:, 0], dateA[:, 1], c=y1)
    plt.show()
    """
    kd树测试
    """
    # dateA, y = make_blobs(n_samples=10000, n_features=2, centers=[[1,1],[7,7],[-2,-3]], random_state=None)
    # tree = kdTree()
    # tree.createKdTree(dateA)
    # k = 1.5
    # point = [3,3]
    # # k = None
    # points2, dis2 = tree.checkRight(dateA, smp=point, k=k)
    # points, dis = tree.searchK(point, k=k)
    #
    # plt.figure()
    # sub1 = plt.subplot(121)
    # sub1.scatter(dateA[:, 0], dateA[:, 1])
    # sub1.scatter(points2[:, 0], points2[:, 1],c='r' )
    # plt.title(u'循环遍历搜索k近邻结果')
    # plt.grid()
    # sub1.scatter(point[0],point[1])
    # sub2 = plt.subplot(122)
    # sub2.scatter(dateA[:, 0], dateA[:, 1])
    # sub2.scatter(points[:, 0], points[:, 1], c='g')
    # plt.title(u'kd树搜索k近邻结果')
    # sub2.scatter(point[0],point[1])
    # plt.grid()
    # plt.show()
    # writer = treeWriter(tree)
    # writer.write()  # write result to tree.png



