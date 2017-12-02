#coding=utf-8
from sklearn import metrics
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from numpy import random

X=np.arange(15).reshape(5,3)
y=np.arange(5)
Y_1 = np.arange(5)
random.shuffle(Y_1)
Y_2 = np.arange(5)
random.shuffle(Y_2)
Y = np.c_[Y_1,Y_2]

def multiclassSVM():
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2,random_state=0)
    model = OneVsRestClassifier(SVC())
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print(predicted)

def multilabelSVM():
    Y_enc = MultiLabelBinarizer().fit_transform(Y)
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y_enc, test_size=0.2, random_state=0)
    model = OneVsRestClassifier(SVC())
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    print(predicted)

def load_data():
    
    
if __name__ == '__main__':
    multiclassSVM()
    #multilabelSVM()