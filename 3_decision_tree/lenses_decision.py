#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from decision_tree_id3 import DecisionTreeID3
from treePlotter import *

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]

lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRte']
lensesTree = DecisionTreeID3.createTree(lenses, lensesLabels)

createPlot(lensesTree)

