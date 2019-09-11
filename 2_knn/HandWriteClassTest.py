#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from . import *
from os import listdir
from numpy import *
from KNN import *
import operator
import matplotlib
import matplotlib.pyplot as plt



def img2vector(filename):
    '''
    将32x32的字符串，转换为1x1024的数组
    :param filename:
    :return:
    '''
    returnVect = zeros((1, 1024))
    fr = open(filename)
    allLineStr = fr.readlines()
    for i in range(32):
        lineStr = allLineStr[i]
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

def handOneFile(prefix,fileNameStr):

    fileName = fileNameStr.split('.')[0]
    label = int(fileName.split('_')[0])
    oneVec = img2vector(prefix % fileNameStr)
    return oneVec,label

def handWritingClassTest():
    hwLabels = []
    trainingFileList = listdir('./digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        oneVec ,classNumStr = handOneFile('./digits/trainingDigits/%s',trainingFileList[i])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = oneVec

    testFileList = listdir('./digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in  range(mTest):
        testVec, testLabel = handOneFile('./digits/testDigits/%s',testFileList[i])
        classifierResult = classify0(testVec,trainingMat,hwLabels,6)
        if (classifierResult != testLabel):
            errorCount += 1

    print(errorCount/float(mTest))

handWritingClassTest()