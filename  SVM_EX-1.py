#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:35:00 2022

@author: Parth Patel
"""

import numpy as np
import pandas as pd
import os

#1.	Load the data into a pandas dataframe named data_firstname where first name is you name.
path = "/Users/sambhu/iCloud Drive (Archive) - 1/Desktop/centennial college /sem-2/Supervised learning 247/Assignment /ass-2-SVM"
filename = 'breast_cancer.csv'
fullpath = os.path.join(path,filename)
data_parth  = pd.read_csv(fullpath)

#2.	Carryout some initial investigations:
print(data_parth.columns.values)
print(data_parth.info())
print(data_parth.dtypes)
print(data_parth.isna().sum())
print(data_parth.mean())
print(data_parth.min())
print(data_parth.max())
print(data_parth.median())
print(data_parth.count())

#3.	Replace the ‘?’ mark in the ‘bare’ column by np.nan and change the type to ‘float’
data_parth['bare'].replace({"?": np.nan},inplace=True)
data_parth = data_parth.astype({'bare': np.float})
print(data_parth.dtypes)

#4.	Fill any missing data with the median of the column.
data_parth['bare'] = data_parth['bare'].fillna(data_parth['bare'].median())

#5.	Drop the ID column
data_parth = data_parth.drop(columns =['ID'])

#6.	Using Pandas, Matplotlib, seaborn (you can use any or a mix) generate 3-5 plots 
#import matplotlib.pyplot as plt
#from pandas.plotting import scatter_matrix

data_parth.hist(figsize=(10,10))

data_parth.plot.box()

data_parth.plot.scatter(x="size", y="shape", s=50);


#7.	Separate the features from the class.

features= ["thickness","size","shape","Marg","Epith","bare","b1","nucleoli","Mitoses"]

x = data_parth[features]
y= data_parth["class"]

#8.	Split your data into train 80% train and 20% test, use the last two digits of your student number for the seed. 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20)
np.random.seed(43)

#9.	Train an SVM classifier using the training data, set the kernel to linear and set the regularization parameter to C= 0.1. Name the classifier clf_linear_firstname.
from sklearn.svm import SVC 

clf_linear_parth = SVC(kernel='linear', C=0.1)

clf_linear_parth.fit(x_train, y_train)

clf_linear_parth.predict(x_test)
print("")
print("--------------------------")
print("Linear SVM:")
print("--------------------------")
#10.	Print out two accuracy score one for the model on the training set i.e. X_train, y_train and the other on the testing set i.e. X_test, y_test. Record 
print("score by training data",clf_linear_parth.score(x_train,y_train))
print("score by testing data",clf_linear_parth.score(x_test,y_test))

#11.	Generate the accuracy matrix
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(clf_linear_parth, x_train, y_train, cv=3)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_train, y_train_pred))

#12.	Repeat steps 9 to 11, in step 9 change the kernel to “rbf” and do not set any value for C.
clf_rbf_parth = SVC(kernel='rbf')

clf_rbf_parth.fit(x_train, y_train)
print("")
print("--------------------------")
print("rbf SVM:")
print("--------------------------")
print("score by training data",clf_rbf_parth.score(x_train,y_train))
print("score by testing data",clf_rbf_parth.score(x_test,y_test))

y_train_pred = cross_val_predict(clf_rbf_parth, x_train, y_train, cv=3)

print(confusion_matrix(y_train, y_train_pred))

#13.	Repeat steps 9 to 11, in step 9 change the kernel to “poly” and do not set any value for C.

clf_poly_parth = SVC(kernel='poly')
print("")
print("--------------------------")
print("Poly SVM:")
print("--------------------------")
clf_poly_parth.fit(x_train, y_train)

print("score by training data",clf_poly_parth.score(x_train,y_train))
print("score by testig data",clf_poly_parth.score(x_test,y_test))

y_train_pred = cross_val_predict(clf_poly_parth, x_train, y_train, cv=3)

print(confusion_matrix(y_train, y_train_pred))

#14.	Repeat steps 9 to 11, in step 9 change the kernel to “sigmoid” and do not set any value for C.
clf_sigmoid_parth = SVC(kernel='sigmoid')

clf_sigmoid_parth.fit(x_train, y_train)
print("")
print("--------------------------")
print("Sigmoid SVM:")
print("--------------------------")
print("score by training data",clf_sigmoid_parth.score(x_train,y_train))
print("score by testing data",clf_sigmoid_parth.score(x_test,y_test))

y_train_pred = cross_val_predict(clf_sigmoid_parth, x_train, y_train, cv=3)

print(confusion_matrix(y_train, y_train_pred))


