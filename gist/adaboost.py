from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm,preprocessing
from sklearn.cross_validation import cross_val_score
import numpy as np
import sys
import collections


gists = np.load('gist.npz')

X = gists['X']
y = gists['y']

clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, X, y)
scores.mean()                             