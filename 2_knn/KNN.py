#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', ' A', ' B', ' B']
    return group, labels


def classify0(row, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 将一行数据扩充至dataSetSize行
    diffMat = tile(row, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    # 无参时，所有全加；
    # axis = 0，按列相加；
    # axis = 1，按行相加；
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    # 返回的是数组值从小到大的索引值
    sortedDistIndex = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndex[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 对value进行倒序排序
    # classCount.items()将classCount字典分解为元组列表
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(fileName):
    '''
    将文件的内容解析成矩阵
    :param fileName:
    :return:
    '''
    fr = open(fileName)
    arrayAllLines = fr.readlines()
    numberOfLines = len(arrayAllLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayAllLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def testClassify0():
    group, labels = createDataSet()
    print(group, labels)
    print(classify0([0.9, 0.8], group, labels, 3))


def plot(datingDataMat, datingLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
               15.0 * array(datingLabels), 15.0 * array(datingLabels))
    plt.show()


def autoNorm(dataSet):
    '''
    数据归一化，数据太大或太小都不好处理
    newVals = (oldVals - minVals)/(maxVals - minVals)
    :param dataSet:
    :return:
    '''
    # 3x1
    minVals = dataSet.min(0)
    # 3x1
    maxVals = dataSet.max(0)
    # 3x1
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 下面是特征值相除
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.05
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :]
                                     , normMat[numTestVecs:m, :]
                                     , datingLabels[numTestVecs:m], 3)
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the tatal error rate is: %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("flier miles?"))
    iceCream = float(input("ice cream per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print("You like this person:",resultList[classifierResult-1])


