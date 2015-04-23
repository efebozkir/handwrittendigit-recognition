# EFE BOZKIR, Technische Universitaet Muenchen

from __future__ import division

__author__ = 'efe'

from mnist import MNIST
from sklearn import tree
from sklearn import metrics
from sklearn import cross_validation
import matplotlib.pyplot as plt
import StringIO, pydot

# Exercise 3.1
mndata = MNIST('./Datasets')
trainingImages, trainingLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

trainingImagesCount = len(trainingImages)
testingImagesCount = len(testImages)

clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=32, max_features=784)
#clf = tree.DecisionTreeClassifier()
clf = clf.fit(trainingImages[:1000], trainingLabels[:1000])
#clf = clf.fit(trainingImages[:60000], trainingLabels[:60000])

predictionRes = clf.predict(testImages)

print metrics.classification_report(testLabels.tolist(), predictionRes, digits=4)

# Cross Validation Results Exercise 3.3 for Decision Tree
scores = cross_validation.cross_val_score(clf, trainingImages[:1000], trainingLabels[:1000].tolist(), cv=5)
print scores
print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

# Pixel importances on 28*28 image
importances = clf.feature_importances_
importances = importances.reshape((28, 28))

# Plot pixel importances
plt.matshow(importances, cmap=plt.cm.hot)
plt.title("Pixel importances for decision tree")
plt.show()

# Decision Tree as output -> decision_tree.png
dot_data = StringIO.StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_png('decision_tree.png')

# IMPORTANT NOTE: If you change the number of training images, you should also change the number of images
# in cross validation.

# decision_tree.png can be huge. Please zoom in to see the tree more clearly.
