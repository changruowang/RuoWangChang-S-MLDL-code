#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np

class KdNode(object):
    def __init__(self,date,left=None,right=None,axis=None):
        self.date = date
        self.axis = axis
        self.left = left
        self.right = right

class kdTree(object):
    def __init__(self):
        self.KdTree = None
        self.depth = 0

    def createKdTree(self, date):

        date = np.array(date)
        k = date.shape[1]

        def create(dateSmp,axis):
            m = dateSmp.shape[0]
            if m == 0: return None
            # axis = np.argmax(np.var(dateSmp,axis=0))
            seqIndex = np.argsort(dateSmp[:,axis])
            dateTmp = dateSmp[seqIndex,:].copy()
            splitPos = len(seqIndex)//2

            leftDate = dateTmp[:splitPos,:]
            rightDate = dateTmp[(splitPos+1):,:]

            node = KdNode(dateTmp[splitPos,:], axis=axis)
            axis = (axis+1)%k
            node.left = create(leftDate,axis)
            node.right = create(rightDate ,axis)

            return node
        self.KdTree = create(date, 0)

    # temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))
    def printTree(self):
        def printX(Node):
            if Node is not None:
                print(Node.date,Node.axis)
                printX(Node.left)
                printX(Node.right)
        printX(self.KdTree)
    @staticmethod
    def calDistance(x_1,x_2):
        return np.sqrt(np.sum((np.array(x_1) - np.array(x_2)) ** 2))

    def searchK(self,smpX,k):
        smpX = np.array(smpX)
        nodeTmp = self.KdTree
        nodeStack = []
        usedNodeList = []
        while nodeTmp is not None:
            nodeStack.append(nodeTmp)
            axisTmp = nodeTmp.axis
            if smpX[axisTmp] > nodeTmp.date[axisTmp]:
                nodeTmp = nodeTmp.right
            else:
                nodeTmp = nodeTmp.left

        if nodeStack[-1].left is not None:    #确定回溯到了叶节点
            nodeStack.append(nodeStack[-1].left)
        elif nodeStack[-1].right is not None:
            nodeStack.append(nodeStack[-1].right)

        nodeTmp = nodeStack.pop(-1)
        shortestDis = self.calDistance(nodeTmp.date, smpX)
        if k is None:
            nearestPoint = nodeTmp.date
        else:
            nearestPoint = []
            if shortestDis < k:
                nearestPoint.append(nodeTmp.date)
            shortestDis = k

        usedNodeList.append(nodeTmp)

        while len(nodeStack)>=1:
            nodeTmp = nodeStack.pop(-1)
            if nodeTmp is None:continue
            usedNodeList.append(nodeTmp)
            distTmp = self.calDistance(nodeTmp.date, smpX)

            if distTmp < shortestDis:
                if k is None:
                    shortestDis = distTmp;  nearestPoint = nodeTmp.date
                else:
                    nearestPoint.append(nodeTmp.date)

            axis = nodeTmp.axis
            if smpX[axis] >= nodeTmp.date[axis]:
                if  nodeTmp.right not in usedNodeList:
                    nodeStack.append(nodeTmp.right)
                if  np.fabs(smpX[axis] - nodeTmp.date[axis]) <= shortestDis:
                    nodeStack.append(nodeTmp.left)
            if smpX[axis] < nodeTmp.date[axis]:
                if  nodeTmp.left not in usedNodeList:
                    nodeStack.append(nodeTmp.left)
                if  np.fabs(smpX[axis] - nodeTmp.date[axis]) <= shortestDis:
                    nodeStack.append(nodeTmp.right)

        return  np.array(nearestPoint), shortestDis

    def checkRight(self, date, smp, k=None):
        m = date.shape[0]
        shortestDis = 100000
        bestDate = None
        if k is None:
            for i in range(m):
                disTmp = self.calDistance(date[i],smp)
                if shortestDis > disTmp:  shortestDis = disTmp; bestDate = date[i]
        else:
            bestDate = []
            for i in range(m):
                disTmp = self.calDistance(date[i], smp)
                if k > disTmp:
                    bestDate.append(date[i])
        return np.array(bestDate), shortestDis


