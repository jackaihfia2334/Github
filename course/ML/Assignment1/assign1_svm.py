# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 16:27:07 2022

@author: myf
"""
#Import Library
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

#加载数据
path1 = "D:/3180104584/MyGit/Github/course/ML/Assignment1/training.data"
path2 = "D:/3180104584/MyGit/Github/course/ML/Assignment1/testing.data"

df1 = pd.read_csv(path1, header=None, sep='\s+')
df2 = pd.read_csv(path2, header=None, sep='\s+')
#print(len(df))

np_train = np.array(df1)
features = np_train[...,:6]
labels = np_train[...,6]
labels = (labels + 1)/2
#print(features[0])


#划分训练集和验证集
from sklearn.model_selection import train_test_split

train_X,test_X, train_y, test_y = train_test_split(features,labels,test_size = 0.1,random_state = 232)


#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 
model = svm.SVC(C=1, kernel='rbf', degree=3, gamma=2, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
model.fit(train_X, train_y)
score1 = model.score(train_X, train_y)
score2 = model.score(test_X, test_y)

import pickle #pickle模块

#保存Model(注:save文件夹要预先建立，否则会报错)
with open('save/model.pickle', 'wb') as f:
    pickle.dump(model, f)

#import pickle #pickle模块
filename = 'test_data.txt'

#读取Model
with open('save/model.pickle', 'rb') as f:
    model = pickle.load(f)
    #测试读取后的Model
    x_test = np.array(df2)
    predicted= model.predict(x_test)
    output = predicted*2-1  #Predict Outputs
    #结果写入txt
    with open(filename,'w') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据
        for a in output:
            #a_s=a.astype(str)
            f.write(str(int(a))+"\n")




    


"""
网格检索参数
parameters={'kernel':['rbf','sigmoid','poly'],'C':np.linspace(0.1,10,5),'gamma':np.linspace(0.1,10,5)}
svc = svm.SVC()
model = GridSearchCV(svc,parameters,cv=5,scoring='accuracy')
model.fit(features, labels)
model.best_params_
model.score(features, labels)
"""