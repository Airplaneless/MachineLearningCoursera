# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:59:58 2017

@author: airplaneless
"""
import pandas
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

data = pandas.read_csv('wine.dat', header=None)

# Set of features
X = np.array([np.array(pandas.Series.tolist(data[i])) for i in data.columns[1:]])
X = X.transpose()
# Set of targets
Y = np.array(pandas.Series.tolist(data[0]))

Xs = scale(X)

def fit(k, X, Y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kNN = KNeighborsClassifier(n_neighbors=k)
    mark = []
    for train_index, test_index in kf.split(X):
        X_train = [X[i] for i in train_index]
        Y_train = [Y[i] for i in train_index]
        X_test = [X[i] for i in test_index]
        Y_test = [Y[i] for i in test_index]
        kNN.fit(X_train, Y_train)
        mark.append(np.mean(cross_val_score(kNN, X_train, Y_train, cv=kf, scoring='accuracy')))
    return np.mean(mark)
    
marks = []
marksS = []
for i in np.linspace(1,50,50):
    marks.append(fit(i, X, Y))
    marksS.append(fit(i, Xs, Y))
    
x1 = np.linspace(1,50,50)
y1 = marks
y1_max = max(marks)
x1_max = x1[marks.index(y1_max)]

x2 = np.linspace(1,50,50)
y2 = marksS
y2_max = max(marksS)
x2_max = x2[marksS.index(y2_max)]

plt.plot(x1,y1)
plt.plot(x1_max, y1_max, '*')

plt.plot(x2,y2)
plt.plot(x2_max, y2_max, '*')

plt.show()

print x2_max
print y2_max