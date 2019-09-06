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


group, labels = createDataSet()
print(group, labels)
