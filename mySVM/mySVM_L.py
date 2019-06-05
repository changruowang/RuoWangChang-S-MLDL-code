# -*- coding: utf-8 -*-
import numpy as np

'''
在第二个变量j/a2的选择上
发现，使用优先更新已经更新过的样本a 计算收敛性较好

比如: 扫描所有样本寻找detaE最大的  与  只扫描标记更新过的a中detaE最大的
      后者算法收敛性较好
对于E 每次寻找j时对应的E都要计算 不能用列表保存的 除非每次参数变化都更新一遍所有的E
'''

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

class SVM():
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
        self.E = np.zeros((self.m, 2))
        self.alpha = np.zeros(self.m)
        self.b = 0.0
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = self.kernel(self.X,self.X[i,:])  #计算初始K核矩阵
        print(self.K)
        # for i in range(self.m):
        #     self.calEi(i)   #计算初始E列表
    """
    self.Y: 一维array     X:二维mat   E:一维array    alpha:一维array   
         K: 二维mat          
    """
    def calEi(self, i):
        gxi = np.mat(np.multiply(self.alpha, self.Y))*self.K[:, i] + self.b
        exi = gxi - float(self.Y[i])
        return exi

    def selectRandJ(self,i):
        j = i  # we want to select any J not equal to i
        while j == i:
            j = int(np.random.uniform(0, self.m))
        return j


    def selectJ(self,i, Ei):
        maxDeltaE = -1
        bestJ = -1
        bestEj = -1

        validE = np.nonzero(self.E[:,0])[0]
        if len(validE) > 0:
            for j in validE:
                if j == i:  continue
                Ej = self.calEi(j)
                if np.fabs(Ei - Ej) > maxDeltaE:
                    bestJ = j;  maxDeltaE = np.fabs(Ei - Ej);  bestEj = Ej
            return bestEj, bestJ
        else:
            j = self.selectRandJ(i)
            Ej = self.calEi(j)
            return Ej, j

    def examineExample(self,i):
        Ei = self.calEi(i)
        r2 = Ei * self.Y[i]

        if (r2<-self.tol) and (self.alpha[i]<self.C) or (r2>self.tol) and (self.alpha[i]>0):
            self.E[i, :] = [1, Ei]
            Ej ,j = self.selectJ(i, Ei)

            alpha1Old = self.alpha[i].copy()
            alpha2Old = self.alpha[j].copy()

            if self.Y[i] == self.Y[j]:
                L, H = np.max([0, alpha2Old + alpha1Old - self.C]), np.min([self.C, alpha2Old + alpha1Old])
            else:
                L, H = np.max([0, alpha2Old - alpha1Old]), np.min([self.C, self.C + alpha2Old - alpha1Old])
            if L == H: print("L==H", alpha2Old, alpha1Old); return 0

            eta = self.K[i, i] + self.K[j, j] - 2 * self.K[i, j]
            if eta <= 0: print("eta<=0"); return 0

            alpha2New = alpha2Old + self.Y[j] * (Ei - Ej) / eta
            alpha2New = clipAlpha(alpha2New, H, L)
            self.alpha[j] = alpha2New

            if np.fabs(alpha2New - alpha2Old) < 0.0000001: print("j not moving enough"); return 0
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
            self.E[j,:] = [1, Ej]
            return 1
        return 0
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
                    print ("non-bound, iter: %d i:%d, pairs changed %d" % (it_cnt, i, numChanged))
                it_cnt += 1
            if examineAll == 1:   examineAll = 0
            elif numChanged == 0: examineAll = 1
            print("iteration number: %d" % it_cnt)

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