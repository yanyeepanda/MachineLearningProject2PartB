__author__ = 'Yanyi'

import numpy as np
from sklearn import mixture
from sklearn import metrics

import cPickle as pickle
# load training data and target label from pickle file
train_data = pickle.load(open('train_feature.p'))
test_data = pickle.load(open('test_feature.p'))
target = pickle.load(open('label.p'))

X_train = np.array(train_data)
Y_train = np.array(target)

# Fit a mixture of Gaussians with EM using 3 components
gmm = mixture.GMM(n_components=3, n_iter=1000)
gmm.fit(X_train)

# predict label for test data
print gmm.predict(test_data)

print Y_train
print gmm.predict(train_data)

# evaluation of the model
# all members of a given class are assigned to the same cluster
print metrics.completeness_score(target, gmm.predict(train_data))
# each cluster contains only members of a single class
print metrics.homogeneity_score(target, gmm.predict(train_data))
# harmonic mean of completeness_score and homogeneity_score
print metrics.v_measure_score(target, gmm.predict(train_data))
# The adjusted Rand index is a function that measures the similarity of the two assignments,
# ignoring permutations and with chance normalization
# Bounded range [-1, 1]: negative values are bad (independent labelings), 1.0 is the perfect match score.
print metrics.adjusted_rand_score(target, gmm.predict(train_data))
# the Mutual Information is a function that measures the agreement of the two assignments
# Bounded range [0, 1]
print metrics.adjusted_mutual_info_score(target, gmm.predict(train_data))


