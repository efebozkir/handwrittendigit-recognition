# EFE BOZKIR, Technische Universitaet Muenchen

from __future__ import division
__author__ = 'efe'

from mnist import MNIST
from sklearn import svm
from sklearn import metrics

# Exercise 1
mndata = MNIST('./Datasets')
trainingImages, trainingLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

trainingImagesCount = len(trainingImages)
testingImagesCount = len(testImages)

#Exercise 2

clf = svm.SVC(kernel='poly', degree = 2)
#clf = svm.SVC(kernel='linear')
#clf = svm.SVC(kernel='rbf')
clf.fit(trainingImages[:1000], trainingLabels[:1000])
#clf.fit(trainingImages[:60000], trainingLabels[:60000])

predictionRes = clf.predict(testImages)

# Calculation of the success of the test phase via metrics
print metrics.classification_report(testLabels.tolist(), predictionRes)

# Manual calculation of the success of the test phase
#rightClassifiedTestImages = 0

#for x in range(testingImagesCount):
#    if(testLabels[x]==clf.predict(testImages[x])[0]):
#        rightClassifiedTestImages+=1

#print rightClassifiedTestImages
#print "Success: %f" %(rightClassifiedTestImages/testingImagesCount)

