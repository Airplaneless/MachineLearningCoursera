# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:41:35 2017

@author: airplaneless
"""

import pandas
import numpy as np
from sklearn.linear_model.perceptron import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

test_data = pandas.read_csv('perceptron-test.csv', header=None)
train_data = pandas.read_csv('perceptron-train.csv', header=None)

Y_train = np.array(pandas.Series.tolist(train_data[0]))
X_train = np.array([np.array(pandas.Series.tolist(train_data[i])) for i in train_data.columns[1:]])
X_train = X_train.transpose()
Y_test = np.array(pandas.Series.tolist(test_data[0]))
X_test = np.array([np.array(pandas.Series.tolist(test_data[i])) for i in test_data.columns[1:]])
X_test = X_test.transpose()

clf = Perceptron(random_state=241)
clf.fit(X_train, Y_train)

val1 = accuracy_score(Y_test, clf.predict(X_test))
print "Unscaled accur.: {}".format(val1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = Perceptron(random_state=241)
clf.fit(X_train, Y_train)

val2 = accuracy_score(Y_test, clf.predict(X_test))
print "Scaled accur.: {}".format(val2)

with open('6.txt', mode='w') as _file:
    _file.write(str(np.around(val2-val1, decimals=3)))