__author__ = 'Yanyi'

import cPickle as pickle

train_data = pickle.load(open('train_feature.p'))
test_data = pickle.load(open("test_feature.p"))
target = pickle.load(open('label.p'))

import numpy as np
x = np.array(train_data)
y = np.array(target)

from sklearn import svm
from sklearn.cross_validation import cross_val_score
clf = svm.SVC()
clf.fit(x, y)

score = cross_val_score(clf, x, y, cv=5)
print score

raw_results = []
for data in test_data:
    data = [int(i) for i in data]
    result = clf.predict([data]).tolist()
    raw_results.append(result)
print raw_results
