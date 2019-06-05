# coding=utf-8
# coding=utf-8
import pandas as pd
import numpy as np


'''
MyLineSearch线性搜索
回溯法搜索a,使梯度下降 x = x - a*gradient更快
L(x - a*gradient) <= L(x) + c*a*gradient*gradient
参数：LossFunc损失函数(入口参数为数据集,x), X,Y数据集,   
'''
class MyLineSearch:
    def __init__(self,X=None, Y=None, LossFunc=None):
        self.alpha = 1
        self.X=X
        self.Y=Y
        self.LossFunc=LossFunc
    def fit(self, theta, gradient):   #theta列向量
        vyGradient = gradient.T
        lossNow = self.LossFunc(X=self.X, Y=self.Y, theta=theta)
        lossNew = self.LossFunc(X=self.X, Y=self.Y, theta=(theta - self.alpha*vyGradient))
        cnt = 1
        c1=0.01
        while lossNow > lossNew and cnt < 100:
            cnt+=1
            self.alpha *= 2.0
            lossNew = self.LossFunc(X=self.X, Y=self.Y, theta=(theta-self.alpha*vyGradient))
        cnt = 100
        while lossNew > (lossNow + c1*gradient*gradient.T*self.alpha) and cnt > 0:
            self.alpha/=2.0
            lossNew = self.LossFunc(X=self.X, Y=self.Y, theta=(theta-self.alpha * vyGradient))
            cnt-=1
        return self.alpha


class MySoftMax():
    def __init__(self, learning_rate=0.01, n_iterations=3000):
        self.theta = None
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.cost = None
        self.label_mounts = 0
        self.sample_mounts = 0
        self.feat_mounts = 0
    def fit(self,X_sample, Y_Sample):
        Y_Sample = np.mat(Y_Sample, dtype=int).reshape(-1,1)
        X = np.insert(np.mat(X_sample), 0, 1, axis=1)
        self.feat_mounts = X.shape[1]
        self.label_mounts = len(np.unique(Y_Sample.tolist()))
        self.sample_mounts = X.shape[0]
        self.cost = np.zeros(self.n_iterations)
        Y = np.zeros((self.sample_mounts, self.label_mounts))
        Y_kind = np.eye(self.label_mounts)
        for i in range(self.sample_mounts):
            Y[i] = Y_kind[Y_Sample[i]]
        self.theta = np.mat(np.ones((self.feat_mounts, self.label_mounts)))
        for i in range(self.n_iterations):
            xyGradient = self.gradient(X, Y)
            self.theta -= xyGradient * self.learning_rate
            # self.cost[i] = self.soft_max_loss(X, Y, self.theta)
    @staticmethod
    def soft_max(z):    #返回列向量
        return np.exp(z)/np.sum(np.exp(z), axis=1)
    def gradient(self, X, Y):   #Y为列向量
        proK = self.soft_max(X*self.theta)
        error = Y - proK
        xyGradient = X.T * error
        return -xyGradient
    def predict(self,X_date):
        X = np.insert(X_date, 0, 1, axis=1)
        y_hat = self.soft_max(X*self.theta)
        predicted = y_hat.argmax(axis=1).getA()
        return predicted


class MyLineRegression():
    def __init__(self, learning_rate=0.005, n_iterations=3000, UseGradient=True):
        self.learning_rate=learning_rate
        self.n_iterations=n_iterations
        self.UseGradient=UseGradient
        self.cost=None
        self.theta=None
    def fit(self, X, Y):
        X = np.insert(X, 0, 1, axis=1)
        thetaNums = X.shape[1]
        sa = MyLineSearch(X=X, Y=Y, LossFunc=self.loss)
        self.theta = np.mat(np.zeros(thetaNums)).T
        if self.UseGradient:
            self.cost = np.zeros(self.n_iterations)
            for i in range(self.n_iterations):
                gradient = self.gradient(X, Y, self.theta)
                alpha = sa.fit(self.theta, gradient)
                # for j in range(thetaNums):
                self.theta -= (alpha * gradient).T
                self.cost[i] = self.loss(X, Y, self.theta)
        else:
            self.theta = (X.T * X).I * X.T * Y
            self.cost = self.loss(X, Y, self.theta)
        # X = np.insert(X, 0, 1, axis=1)
        # sampleNums = X.shape[0]
        # thetaNums = X.shape[1]
        # self.cost=np.zeros(self.n_iterations * sampleNums)
        # self.theta = np.mat(np.zeros(thetaNums)).T
        # if self.gradient:
        #     for i in range(self.n_iterations):
        #         for k in range(sampleNums):
        #             for j in range(thetaNums):
        #                 derivativeInner = X[k, :] * self.theta - Y[k, 0]
        #                 self.theta[j, 0] = self.theta[j, 0] - derivativeInner * X[k, j] * self.learning_rate
        #             self.cost[i * sampleNums + k] = cost_func(X, Y, self.theta.T)

    @staticmethod
    def loss(X, Y, theta):
        inner = np.power((X * theta) - Y, 2)
        return np.sum(inner) / (2 * len(X))
    @staticmethod
    def gradient(X, Y, theta):
        error = (X * theta) - Y
        intThetaNum = int(X.shape[1])
        xvGradient = np.mat(np.zeros(intThetaNum))
        for j in range(intThetaNum):
            xvGradient[0, j] = np.sum(np.multiply(error, X[:, j])) / (len(X))
        return xvGradient
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(self.theta)
    def Theta(self):
        return self.theta
    def costN(self):
        return self.cost
