__author__ = 'Yanyi'

import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
import cPickle as pickle

# load training data and target label from pickle file
train_data = pickle.load(open('train_feature.p'))
test_data = pickle.load(open('test_feature.p'))
target = pickle.load(open('label.p'))

X_train = np.array(train_data)
Y_train = np.array(target)

# build up model using training data
k_means = KMeans(init='k-means++', n_clusters=3, n_init=200, max_iter=2000)
k_means.fit(X_train)

print Y_train
print k_means.labels_

# evaluation of the model
# all members of a given class are assigned to the same cluster
print metrics.completeness_score(target, k_means.labels_)
# each cluster contains only members of a single class
print metrics.homogeneity_score(target, k_means.labels_)
# harmonic mean of completeness_score and homogeneity_score
print metrics.v_measure_score(target, k_means.labels_)
# The adjusted Rand index is a function that measures the similarity of the two assignments,
# ignoring permutations and with chance normalization
# Bounded range [-1, 1]: negative values are bad (independent labelings), 1.0 is the perfect match score.
print metrics.adjusted_rand_score(target, k_means.labels_)
# the Mutual Information is a function that measures the agreement of the two assignments
# Bounded range [0, 1]
print metrics.adjusted_mutual_info_score(target, k_means.labels_)

# predict labels of the test data
print k_means.predict(test_data)


# draw 2D plot of kmeans
from os import listdir
dirs = listdir('data')
suburb = []
for file in dirs:
    suburb.append(file[:-11])

from sklearn.decomposition import PCA
import pylab as pl

pca = PCA(n_components=2).fit(X_train)
pca_2d = pca.transform(X_train)

fig, ax = pl.subplots()
for s, x, y in zip(suburb,pca_2d[:,0],pca_2d[:,1]):
    ax.annotate(s, xy=(x, y), xytext=(-10,10),
            textcoords='offset points', ha='center', va='bottom',
            #bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
            # arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', color='red'),
            fontsize=7
            )
for i in range(0, pca_2d.shape[0]):
    if k_means.labels_[i] == 1:
        c1 = ax.scatter(pca_2d[i,0],pca_2d[i,1],c='r',marker='+')
    elif k_means.labels_[i] == 0:
        c2 = ax.scatter(pca_2d[i,0],pca_2d[i,1],c='g',marker='o')
    elif k_means.labels_[i] == 2:
        c3 = ax.scatter(pca_2d[i,0],pca_2d[i,1],c='b',marker='*')

pl.title('K-means clusters')

pl.show()
