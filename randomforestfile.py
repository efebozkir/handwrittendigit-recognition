# EFE BOZKIR, Technische Universitaet Muenchen

from __future__ import division
__author__ = 'efe'

from mnist import MNIST
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import cross_validation
import matplotlib.pyplot as plt

# Exercise 3.2
mndata = MNIST('./Datasets')
trainingImages, trainingLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

trainingImagesCount = len(trainingImages)
testingImagesCount = len(testImages)

clf = RandomForestClassifier(n_estimators=150, criterion="gini", max_depth=32, max_features="auto")
#clf = RandomForestClassifier()
clf = clf.fit(trainingImages[:1000], trainingLabels[:1000])
#clf = clf.fit(trainingImages[:60000], trainingLabels[:60000])

predictionRes = clf.predict(testImages)

print metrics.classification_report(testLabels.tolist(), predictionRes)

# Cross Validation Results Exercise 3.3 for Random Forests
scores = cross_validation.cross_val_score(clf, trainingImages[:1000], trainingLabels[:1000].tolist(), cv=20)
print scores
print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

# Pixel importances on 28*28 image
importances = clf.feature_importances_
importances = importances.reshape((28, 28))

# Plot pixel importances
plt.matshow(importances, cmap=plt.cm.hot)
plt.title("Pixel importances for random forests")
plt.show()

# IMPORTANT NOTE: If you change the number of training images, you should also change the number of images
# in cross validation.
