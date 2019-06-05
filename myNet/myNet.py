import numpy as np
import sys
sys.path.append('../')
from myTool.myTools import ind2sub
import copy

def tanh(x):
    return np.tanh(x)
def tanh_grad(x):
    return 1 - np.tanh(x)*np.tanh(x)
def logistic(x):
    return 1/(1 + np.exp(-x))
def logistic_grad(x):
    return logistic(x)*(1-logistic(x))


def costFunction(y_hat, Y):
    m = Y.shape[1]
    cost = -np.sum(np.multiply(Y, np.log(y_hat)) + np.multiply(1-Y, np.log(1-y_hat)))/m
    return np.squeeze(cost)

class MyNeuralNet:
    def __init__(self, layers=None, hid_active = 'tanh', out_active = 'logistic'):
        assert (layers.ndim == 1)
        self.loss = None
        self.Layers = np.insert(np.array(layers), 0, 2, axis=0)
        if hid_active == 'tanh':
            self.a1 = tanh
            self.grad_a1 = tanh_grad
        elif hid_active == 'logistic':
            self.a1 = logistic
            self.grad_a1 = logistic_grad
        if out_active == 'tanh':
            self.a2 = tanh
            self.grad_a2 = tanh_grad
        elif out_active == 'logistic':
            self.a2 = logistic
            self.grad_a2 = logistic_grad
        self.cache = []
        self.W = []
        self.b = []
        self.dW = []
        self.db = []
    def paramInit(self,X):
        self.Layers[0] = X.shape[0]
        layerTmp = self.Layers
        np.random.seed(1)
        for i in range (1,len(self.Layers)):
            self.W.append(np.random.randn(layerTmp[i], layerTmp[i-1])*0.01)
            self.b.append(np.zeros((layerTmp[i],1)))
            self.db.append(np.zeros((layerTmp[i], 1)))
            self.dW.append(np.zeros((layerTmp[i], layerTmp[i-1])))
    #执行一次前向传播
    def model_forward(self, A, W, b):
        L = len(W)
        cache = []
        for l in range(0, L):
            assert (A.shape[0] == W[l].shape[1])
            Z = np.dot(W[l], A) + b[l]
            cache.append((Z, A))
            if l != L-1:
                A = self.a1(Z)
            else:
                A = self.a2(Z)
        return A, cache
    def model_backward(self, y_hat, Y, cache):
        m = Y.shape[1]
        L = len(self.W)
        dA = -np.divide(Y, y_hat) + np.divide(1-Y, 1-y_hat)
        for l in reversed(range(L)):
            Z, A = cache.pop()
            if l != L-1:
                dZ = np.multiply(dA, self.grad_a1(Z))
            else:
                dZ = np.multiply(dA, self.grad_a2(Z))
            self.dW[l] = np.dot(dZ, A.T)/m
            self.db[l] = np.sum(dZ, axis=1, keepdims=True)/m
            dA = np.dot(self.W[l].T, dZ)

    def calEdgeJ(self, X, Y, Wb, index, l, delta):
        W = copy.deepcopy(self.W)
        b = copy.deepcopy(self.b)
        Wb[ind2sub(Wb.shape, index)] = Wb[ind2sub(Wb.shape, index)] + delta
        Wl, bl = np.split(Wb, [Wb.shape[1] - 1], axis=1)
        W[l] = Wl
        b[l] = bl
        y_hat, NoUse = self.model_forward(X, W, b)
        return costFunction(y_hat, Y)

    def grad_check(self, X, Y, W, b):
        L = len(W)
        dWb = []
        for l in range(0, L):
            Wbl = np.concatenate([W[l], b[l]], axis=1)
            dWb1 = np.zeros_like(Wbl)
            for i in range(len(Wbl.ravel())):
                J1 = self.calEdgeJ(X, Y, Wbl, i, l, 0.001)
                J2 = self.calEdgeJ(X, Y, Wbl, i, l, -0.001)
                dWb1[ind2sub(Wbl.shape, i)] = (J1 - J2) / 0.002
            dWb.append(dWb1)
        return dWb

    def predict(self,X):
        if X.shape[0] != self.W[0].shape[1]:
            X = X.T
        assert(X.shape[0] == self.W[0].shape[1])
        y_hat, NoUse = self.model_forward(X, self.W, self.b)
        y_hat = y_hat.ravel()
        y_hat[np.where(y_hat > 0.5)] = 1
        y_hat[np.where(y_hat <= 0.5)] = 0

        return y_hat

    def nn_fit(self, X, Y, n_iterations=1000, PrintCost=False, learningRate = 0.075,GradCheck=False, store_cost = True):
        Y = Y.reshape(1,-1)
        if Y.shape[1] != X.shape[1]:
            X = X.T
        assert (Y.shape[1]==X.shape[1])
        self.paramInit(X)
        if store_cost:
            self.loss = np.zeros(n_iterations)
        L = len(self.W)
        for i in range (0, n_iterations):
            y_hat, cache = self.model_forward(X, self.W, self.b)
            self.model_backward(y_hat, Y, cache)
            cost = costFunction(y_hat, Y)
            if PrintCost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))
            if store_cost:
                self.loss[i] = cost
            if GradCheck:
                ndWb = self.grad_check(X,Y,self.W,self.b)
                for l in range(0,L):
                    Wb1 = np.concatenate([self.dW[l], self.db[l]], axis=1)
                    delta = np.abs(ndWb[l] - Wb1)
                    if len(delta[delta > 0.01]) != 0:
                        print(delta)
                        assert (len(delta[delta > 0.01]) == 0)
                    # print(delta)
                    # assert (len(delta[delta > 0.01]) == 0)
            for l in range(0,L):
                self.W[l] = self.W[l] - learningRate * self.dW[l]
                self.b[l] = self.b[l] - learningRate * self.db[l]

    def getLoss(self):
        return self.loss
    def printParam(self):
        # print('X=' + str(X_input) + '\n')
        assert (type(self.W) == type([]))
        assert (len(self.W) == len(self.b))
        L = len(self.W)
        for l in range(0, L):
            str_w = 'W' + str(l + 1) + ':' + str(self.W[l]) + '\n'
            str_b = 'b' + str(l + 1) + ':' + str(self.b[l])
            print(str_w + str_b)
        return self.W, self.b