# -*- coding: utf-8 -*-

from math import log

"""
计算香浓熵 
参数：数据集 以及所要求的数据集的列的熵的列号
"""
def calShannonEntropy(DataSet):
    entropy = 0.0
    listDataColumn = [sample[-1] for sample in DataSet]
    intDataMount = len(listDataColumn)
    setData = set(listDataColumn)
    dictTypeCount = {}
    for keys in setData:
        dictTypeCount.update({keys: listDataColumn.count(keys)})
    for keys in dictTypeCount.keys():
        pro = float(dictTypeCount[keys]) / intDataMount
        entropy -= pro * log(pro, 2)
    return entropy


"""

"""
def deleteOneDate(smp,axis):
    listTmp = smp[:]
    listTmp.pop(axis)
    return listTmp

def spitDispersedData(DataSet, axis, Value):
    listSubData = [deleteOneDate(smp,axis) for smp in DataSet if smp[axis] == Value]
    return listSubData

def spitLowerContinueDate(DataSet, axis, Value):
    listSubData = [deleteOneDate(smp,axis) for smp in DataSet if smp[axis] <= Value]
    return listSubData

def spitHigherContinueDate(DataSet, axis, Value):
    listSubData = [deleteOneDate(smp,axis) for smp in DataSet if smp[axis] > Value]
    return listSubData

"""
计算集合D的经验熵H(D)
for ...遍历所有特征
    for ...遍历某个特征中的所有取值情况
        求该特征某一种情况时的条件熵 求和
    得到g(D,A) = H(D) - H(D|A)
    计算H(A),信息增益率 取最大
"""


def selBestSplitFeature(DataSet):
    floatEntropyD = calShannonEntropy(DataSet)  # H(D)
    floatBestInfoGain = 0.0
    floatBestValue = 0.0
    intBestFeature = -1
    intFeatureMounts = len(DataSet[0]) - 1
    for i in range(intFeatureMounts):
        listFeatureI = [item[i] for item in DataSet]
        floatEntropyDA = 0.0
        floatEntropyA = 0.0

        if type(listFeatureI[0]).__name__ == 'float' or type(listFeatureI[0]).__name__ == 'int':
            listFeatureHalf = []
            listFeatureTmp = sorted(list(set(listFeatureI)))
            for j in range(len(listFeatureTmp) - 1):
                listFeatureHalf.append((listFeatureTmp[j] + listFeatureTmp[j+1])/2.0)
            for j in range(len(listFeatureHalf)):
                listLowSubDate = spitLowerContinueDate(DataSet, i, listFeatureHalf[j])
                listHighSubDate = spitHigherContinueDate(DataSet, i, listFeatureHalf[j])
                floatProL = float(len(listLowSubDate)) / float(len(DataSet))
                floatProH= float(len(listHighSubDate)) / float(len(DataSet))
                floatEntropyDA = calShannonEntropy(listLowSubDate) * floatProL + calShannonEntropy(listHighSubDate) * floatProH
                floatEntropyA = -floatProL*log(floatProL, 2) - floatProH*log(floatProH, 2)
                if floatEntropyA == 0:
                    continue
                floatInfoGain = (floatEntropyD - floatEntropyDA) / floatEntropyA  # H(A)
                if floatInfoGain > floatBestInfoGain:
                    floatBestInfoGain = floatInfoGain
                    intBestFeature = i
                    floatBestValue = listFeatureHalf[j]
        else:
            setFeatureI = set(listFeatureI)

            for value in setFeatureI:
                listSubData = spitDispersedData(DataSet, i, value)
                pro = float(len(listSubData)) / float(len(DataSet))
                floatEntropyDA += calShannonEntropy(listSubData) * pro
                floatEntropyA += -pro * log(pro, 2)
            if floatEntropyA == 0:
                continue
            floatInfoGain = (floatEntropyD - floatEntropyDA) / floatEntropyA  # H(A)
            if floatInfoGain > floatBestInfoGain:
                floatBestInfoGain = floatInfoGain
                intBestFeature = i
    return intBestFeature, floatBestValue


"""

"""


def getSetValue(a):
    return a[1]


def selMajorityVote(DataSet):
    listDate = [smp[-1] for smp in DataSet]
    setDateKind = set(listDate)
    dictKindCount = {}
    for item in setDateKind:
        dictKindCount.update({item: listDate.count(item)})
    listSorted = sorted(dictKindCount.items(), key=getSetValue, reverse=True)
    return listSorted[0][0]


"""
如果只有一种标签  返回这个值 叶子
如果特征用完了 多数表决
选择最优分裂特征

DataSet类型为list 支持连续数据 不支持减枝
"""
def createTree(DataSet, Labels):
    listResult = [smp[-1] for smp in DataSet]
    if listResult.count(listResult[0]) == len(listResult):
        return listResult[0]
    if len(DataSet[0]) == 1:
        return selMajorityVote(DataSet)
    intFeatureI, floatValue = selBestSplitFeature(DataSet)
    listFeatureN = [smp[intFeatureI] for smp in DataSet]
    strKey = Labels[intFeatureI]
    dictTreeN = {strKey: {}}
    Labels.remove(strKey)

    if type(listFeatureN[0]).__name__ == 'float' or type(listFeatureN[0]).__name__ == 'int':
        strSubLabels = Labels[:]
        dictTreeN[strKey]['<'+str(floatValue)] = createTree(spitLowerContinueDate(DataSet, intFeatureI, floatValue), strSubLabels)
        strSubLabels = Labels[:]
        dictTreeN[strKey]['>'+str(floatValue)] = createTree(spitHigherContinueDate(DataSet, intFeatureI, floatValue), strSubLabels)
    else:
        setFeatureN = set(listFeatureN)
        for i in setFeatureN:
            strSubLabels = Labels[:]
            dictTreeN[strKey][i] = createTree(spitDispersedData(DataSet, intFeatureI, i), strSubLabels)
    return dictTreeN


"""
"""
def classyDateByTree(Date, dictTree, Labels):
    firstLabel = list(dictTree.keys())
    intFeatIndex = Labels.index(firstLabel[0])
    dictSecondTree = dictTree[firstLabel[0]]

    if type(Date[intFeatIndex]).__name__ != 'str':
        floatSpitValue = float(list(dictSecondTree.keys())[0].strip('>').strip('<'))
        if Date[intFeatIndex] > floatSpitValue:
            item = '>'+str(floatSpitValue)
        else:
            item = '<'+str(floatSpitValue)
        if type(dictSecondTree[item]).__name__ == 'dict':
            return classyDateByTree(Date, dictSecondTree[item], Labels)
        else:
            return dictSecondTree[item]
    else:
        print(Date[intFeatIndex])
        for item in dictSecondTree.keys():
            if Date[intFeatIndex] == item:
                if type(dictSecondTree[item]).__name__ == 'dict':
                    return classyDateByTree(Date, dictSecondTree[item], Labels)
                else:
                    return dictSecondTree[item]


def classifyAll(inputTree, featLabels, testDateSet):
    """
    输入：决策树，分类标签，测试数据集
    输出：决策结果
    描述：跑决策树
    """
    classLabelAll = []
    for testVec in testDateSet:
        classLabelAll.append(classyDateByTree(testVec, inputTree, featLabels))
    return classLabelAll
