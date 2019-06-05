#!/usr/bin/python
# -*- coding:utf-8 -*-
import sys
import numpy as np
from numpy import *
from myTool.myTools import ind2sub
import copy
import os
import matplotlib.pylab as plt

def test(a):
    print(a)
    a = 1
    print(a)

a = np.array([3, 1, 2])
b = np.array([5, 5]).reshape(1,2)
c = np.insert(a, 0, 1, axis=0).reshape(2,-1)
# c = c.ravel()
re = (c < 3)
d = 5
test(d)
print(d)

# print(c,re)
# print(c[c < 3])

# Wb = np.concatenate([c, b.T], axis=1)
# print(Wb)
# c1, b1 = np.split(Wb, [Wb.shape[1]-1], axis=1)
# print(c1,b1)
# d = [1,2,3]
#
# print(c)
# print(np.sum(c, axis=1, keepdims=True))
# for i in reversed(range(4)):
#     print(i)
# print(i)
# print(c)
# print(c[0])
# print(len(c))
#
# print a
# print np.array(matA)[0],matA[1]
# print np.sum(np.power(matA - matB,2),axis=1)
# print type(matA)
# print matA.A - matB.A
# # alpha0C = (1<alpha<3)
# # a = np.argsort(~alpha0C)
# print alpha

