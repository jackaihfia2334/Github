# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 11:27:26 2022

@author: myf
"""
#Import Library
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

path1 = "C:/Users/myf/Downloads/Assignment1/training.data"
path2 = "C:/Users/myf/Downloads/Assignment1/testing.data"

df1 = pd.read_csv(path1, header=None, sep='\s+')
df2 = pd.read_csv(path2, header=None, sep='\s+')
#print(len(df))

np_tests = np.array(df1)
features = np_tests[...,:6]
labels = np_tests[...,6]
labels = (labels + 1)/2
#print(features[0])
"""
for i, test in enumerate(np_tests):
    print("第%d行的数据是%s:" % (i, test))
"""

#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
svc = svm.SVC(C=1, kernel='', degree=3, gamma=2, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
#model.fit(features, labels)
#score = model.score(features, labels)
#Predict Outputs
#x_test = np.array(df2)
#predicted= model.predict(x_test)

parameters={'kernel':['rbf','sigmoid','poly'],'C':np.linspace(0.1,10,5),'gamma':np.linspace(0.1,10,5)}
svc = svm.SVC()
model = GridSearchCV(svc,parameters,cv=5,scoring='accuracy')
model.fit(features, labels)
model.best_params_
model.score(features, labels)