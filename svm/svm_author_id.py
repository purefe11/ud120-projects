#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

from sklearn.svm import SVC
clf = SVC(kernel='rbf', C=10000.0)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print("accuracy : {}".format(accuracy))

from sklearn.metrics import classification_report
print(classification_report(labels_test, pred, target_names=['Sara', 'Chris']))

from sklearn import metrics
print(metrics.confusion_matrix(labels_test,pred))

print("pred 10: {}, pred 26: {}, pred 50: {}".format(pred[10], pred[26], pred[50]))


#########################################################


