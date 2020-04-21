# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:39:06 2020

@author: yanka
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dfTrain = pd.read_csv(r'E:\Machine Learning\COVID-19\randomtrain.csv')
dfTest = pd.read_csv(r'E:\Machine Learning\COVID-19\randomtest.csv')
dfTrain.fillna(dfTrain.mean(),inplace=True)
dfTest.fillna(dfTest.mean(),inplace=True)

#%%
def Target(onerow):
    if onerow['loan_status']=='Fully Paid':
        return 1
    else:
        return 0
dfTrain['target']=dfTrain.apply(Target,axis=1)
dfTest['target']=dfTrain.apply(Target,axis=1)
dfTrain.drop(['loan_status'], axis=1, inplace=True)
dfTest.drop(['loan_status'], axis=1, inplace=True)
#%%
#X_train=dfTrain.drop(['target'], axis=1, inplace=False)
#y_train=dfTrain['target']
#
#X_test=dfTest.drop(['target'], axis=1, inplace=False)
#y_test=dfTest['target']
X_train=dfTrain.drop(['loan_status'], axis=1, inplace=False)
y_train=dfTrain['loan_status']

X_test=dfTest.drop(['loan_status'], axis=1, inplace=False)
y_test=dfTest['loan_status']
#%%
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))   
        
    elif train==False:
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))    
#%%
#from sklearn.model_selection import train_test_split
#X_train, X_val, y_train, y_val = train_test_split(dfTrain.drop('loan_status',axis=1),dfTrain['loan_status'],test_size=0.15,random_state=101)
#%%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)
#%%Oversampling only the training set using Synthetic Minority Oversampling Technique
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=12)
x_train_r, y_train_r = sm.fit_sample(X_train, y_train)

#%%
#logistic Regression
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(C = 0.005,random_state=10)

log_reg.fit(x_train_r, y_train_r)

print_score(log_reg, x_train_r, y_train_r, X_test, y_test, train=False)
#%%
#RandomForest
#from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
clf_rf = RandomForestClassifier(n_estimators=40, random_state=21,min_samples_split=4)
clf_rf.fit(x_train_r, y_train_r)
print_score(clf_rf, x_train_r, y_train_r, X_test, y_test, train=True)