#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from math import log
import operator
import matplotlib.pyplot as plt
import pickle


class DecisionTreeID3(object):

    @staticmethod
    def __calcShannonEnt(dataSet):
        '''
        calc dataSet shannon entropy
        :return:
        '''
        numEntries = len(dataSet)
        labelCount = {}
        for entry in dataSet:

            label = entry[-1]
            if label not in labelCount.keys():
                labelCount[label] = 0

            labelCount[label] += 1
        shannonEnt = 0.0
        for key in labelCount.keys():
            prob = float(labelCount[key] / numEntries)
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt

    # 将数据集按照某一特征分类成多个数据集
    @staticmethod
    def __splitDataSet(dataSet, axis, value):
        splitDataSets = []
        for featVec in dataSet:
            if featVec[axis] == value:
                reduceFeatVec = featVec[:axis]
                reduceFeatVec.extend(featVec[axis + 1:])
                splitDataSets.append(reduceFeatVec)
        return splitDataSets

    # 计算信息增益
    @staticmethod
    def __chooseBestInformationGain(dataSet):
        numFeatures = len(dataSet[0]) - 1
        baseEntropy = DecisionTreeID3.__calcShannonEnt(dataSet)
        baseInfoGain = 0.0;
        bestFeature = -1
        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)
            newEntropy = 0.0
            for uniqueVal in uniqueVals:
                subDataSet = DecisionTreeID3.__splitDataSet(dataSet, i, uniqueVal)
                prob = float(len(subDataSet)) / len(dataSet)
                newEntropy += prob * DecisionTreeID3.__calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy;
            if infoGain > baseInfoGain:
                baseInfoGain = infoGain
                bestFeature = i
        return bestFeature

    @staticmethod
    def __majorityCnt(self, classList):
        '''
        如果切分数据集到最后只剩一个特征时，哪个属性值对应的样本多，则选择对应属性值对应的label
        :param classList:
        :return:
        '''
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount

    @staticmethod
    def createTree(dataSet, featuresLabel):
        '''
        创建决策树
        :param dataSet:
        :param featuresLabel:
        :return:
        '''
        classList = [item[-1] for item in dataSet]
        # 如果数据的分类都相同则直接返回
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        # 如果切分数据集到最后只剩一个特征时，哪个属性值对应的
        # 样本多 ，则选择对应属性值对应的label
        if len(dataSet[0]) == 1:
            return DecisionTreeID3.__majorityCnt(classList)
        bestFeat = DecisionTreeID3.__chooseBestInformationGain(dataSet)
        bestFeatLabel = featuresLabel[bestFeat]
        myTree = {bestFeatLabel: {}}
        del (featuresLabel[bestFeat])
        featValues = [item[bestFeat] for item in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            # 复制特征标签
            subLabels = featuresLabel[:]
            # 每个特征的属性对应一个子树
            myTree[bestFeatLabel][value] = DecisionTreeID3.createTree(
                DecisionTreeID3.__splitDataSet(dataSet, bestFeat, value), subLabels)
        return myTree

    @staticmethod
    def classify(decisionTree, featLabel, vector):
        '''
        给定样本判断其类别
        :param decisionTree:
        :param featLabel:
        :param vector:
        :return:
        '''
        firstStr = list(decisionTree.keys())[0]
        secondDict = decisionTree[firstStr]
        featIndex = featLabel.index(firstStr)
        classLabel = None
        for key in secondDict.keys():
            if vector[featIndex] == key:
                if type(secondDict[key]).__name__ == 'dict':
                    classLabel = DecisionTreeID3.classify(secondDict[key], featLabel, vector)
                else:
                    classLabel = secondDict[key]
        return classLabel

    @staticmethod
    def storeTree(decisionTree, fileName):
        '''
        持久化决策树
        :return:
        '''
        fw = open(fileName, 'w')
        pickle.dump(decisionTree, fw)
        fw.close()

    @staticmethod
    def grabTree(fileName):
        '''
        从持久化决策树中恢复
        :return:
        '''
        fw = open(fileName)
        return pickle.load(fileName)
