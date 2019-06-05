# -*- coding: utf-8 -*-
import numpy as np
from numpy import *
#高斯核函数
class RBF(object):
    def __init__(self, gamma=1):
        self.gamma = gamma
        if gamma == 0:
            self.gamma = 1
    def __call__(self, x, xi):
        # print x,xi
        X_sum = np.sum(np.power(x - xi,2),axis=1)
        return np.exp(X_sum / -2 * self.gamma ** 2)

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

class SVM:
    def __init__(self, C=1, tol=1e-6, kernel=None):
        self.C = C
        self.tol = tol
        self.kernel = kernel
        self.m = None
        self.alpha = None
        self.w = None
        self.E = None
        self.K = None;
        self.X = None;self.IndSV = None
        self.Y = None;self.b = None
    def initParam(self,X, Y):
        self.X = np.mat(X)
        self.Y = np.array(Y).ravel()
        self.m, intFeatMounts = self.X.shape
        self.w = np.mat(np.zeros(intFeatMounts)).reshape(1, -1)
        self.E = np.zeros(self.m)
        self.alpha = np.zeros(self.m)
        self.b = 0.0
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = self.kernel(self.X,self.X[i,:])  #计算初始K核矩阵
        print(self.K)
        # for i in range(self.m):
        #     self.E[i] = self.calEi(i)   #计算初始E列表
        # print self.E
    """
    self.Y: 一维array     X:二维mat   E:一维array    alpha:一维array   
         K: 二维mat          
    """
    def calEi(self, i):
        gxi = np.mat(np.multiply(self.alpha, self.Y))*self.K[:, i] + self.b
        exi = gxi - float(self.Y[i])
        return exi

    def selectJ(self,i, Ei):
        maxDeltaE = 0
        bestJ = -1
        bestEj = 0
        for j in range(self.m):
            if j == i:  continue
            Ej = self.calEi(j)
            if np.fabs(Ei - Ej) > maxDeltaE:
                bestJ = j;  maxDeltaE = np.fabs(Ei - Ej); bestEj=Ej
        return bestEj,bestJ


    def examineExample(self,i):
        Ei = self.calEi(i)
        print(self.alpha[i],Ei)
        if 0 < self.alpha[i] < self.C and abs(self.Y[i] * Ei) < self.tol:
            return 0
        elif self.alpha[i] == 0 and self.Y[i] * Ei > -self.tol:
            return 0
        elif self.alpha[i] == self.C and self.Y[i] * Ei <= self.tol:
            return 0
        else:
        # if (r2<-self.tol) and (self.alpha[i]<self.C) or (r2>self.tol) and (self.alpha[i]>0):
            Ej, j = self.selectJ(i, Ei)

            alpha1Old = self.alpha[i].copy()
            alpha2Old = self.alpha[j].copy()

            if self.Y[i] == self.Y[j]:
                L = max(0, alpha2Old + alpha1Old - self.C)
                H = min(self.C, alpha2Old + alpha1Old)
            else:
                L = max(0, alpha2Old - alpha1Old)
                H = min(self.C, self.C + alpha2Old - alpha1Old)

            if L == H: print("L==H", alpha2Old, alpha1Old); return 0

            eta = self.K[i, i] + self.K[j, j] - 2.0 * self.K[i, j]
            if eta <= 0: print("eta<=0"); return 0
            alpha2New = alpha2Old + self.Y[j] * (Ei - Ej) / eta
            alpha2New = clipAlpha(alpha2New, H, L)
            self.alpha[j] = alpha2New
            if np.fabs(alpha2New - alpha2Old) < 0.00000000001: print("j not moving enough"); return 0
            alpha1New = alpha1Old + self.Y[i] * self.Y[j] * (alpha2Old - alpha2New)
            self.alpha[i] = alpha1New

            b1 = self.b-Ei-self.Y[i]*self.K[i, i]*(alpha1New-alpha1Old)-self.Y[j]*self.K[j, i]*(alpha2New-alpha2Old)
            b2 = self.b-Ej-self.Y[i]*self.K[i, j]*(alpha1New-alpha1Old)-self.Y[j]*self.K[j, j]*(alpha2New-alpha2Old)

            if 0 < alpha1New < self.C:
                self.b = b1
            elif 0 < alpha2New < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            print(i, j)
            return 1
        # return 0
    '''
    初始化 alpha[m] = 0 k[m,m] E[m] b      
    选择alpha 1 2
    先检测alpha > 0 < c 的样本
    '''
    def fit(self,X_Smp, Y_Smp):
        self.initParam(X_Smp,Y_Smp)
        it_cnt = 0
        examineAll = 1
        numChanged = 0
        while it_cnt<50 and (numChanged > 0 or examineAll):
            numChanged = 0
            if examineAll:
                for i in range(self.m):
                    numChanged += self.examineExample(i)
                    print("fullSet, iter: %d i:%d, pairs changed %d" % (it_cnt, i, numChanged))
                it_cnt += 1
            else:
                nonBoundIs = np.nonzero((self.alpha>0) * (self.alpha<self.C))[0]
                for i in nonBoundIs:
                    numChanged += self.examineExample(i)
                    print("non-bound, iter: %d i:%d, pairs changed %d" % (it_cnt, i, numChanged))
                it_cnt += 1
            if examineAll == 1:   examineAll = 0
            elif numChanged == 0: examineAll = 1
            print("iteration number: %d" % it_cnt)


        print(self.alpha)
        self.IndSV = np.nonzero(self.alpha>0)

    def predict(self,X_Smp):
        X_Smp = np.mat(X_Smp)
        m,n = X_Smp.shape
        y_hat = np.zeros(m)
        SVs = self.X[self.IndSV]
        LabelSV = self.Y[self.IndSV]
        for i in range(m):
            kernelEval = self.kernel(SVs,X_Smp[i,:])
            predict = np.mat(np.multiply(self.alpha[self.IndSV],LabelSV))*kernelEval + self.b
            y_hat[i] = np.sign(predict)
        return y_hat