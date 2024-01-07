#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 23:01:34 2022

@author: Parth Patel
"""

import numpy as np
import pandas as pd
import os

#1.	Load the data into a pandas dataframe named data_firstname where first name is you name.
path = "/Users/sambhu/Desktop/sem-2/Supervised learning 247/Assignment /ass-2-SVM"
filename = 'breast_cancer.csv'
fullpath = os.path.join(path,filename)
data_parth  = pd.read_csv(fullpath)

#	Carryout some initial investigations:
print(data_parth.columns.values)
print(data_parth.dtypes)
print(data_parth.isna().sum())
print(data_parth.mean())
print(data_parth.min())
print(data_parth.max())
print(data_parth.median())
print(data_parth.count())

#2.	Replace the ‘?’ mark in the ‘bare’ column by np.nan and change the type to ‘float’
data_parth['bare'].replace({"?": np.nan},inplace=True)
data_parth = data_parth.astype({'bare': np.float})
print(data_parth.dtypes)

#3.	Drop the ID column
data_parth = data_parth.drop(columns =['ID'])

#4.	Separate the features from the class.
features= ["thickness","size","shape","Marg","Epith","bare","b1","nucleoli","Mitoses"]
x = data_parth[features]
y= data_parth["class"]

#5.	Split your data into train 80% train and 20% test, use the last two digits of your student number for the seed. 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20)
np.random.seed(43)

# 6,7.	Combine the two transformers into a pipeline name it num_pipe_firstname.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#imputer = SimpleImputer(missing_values=np.nan, strategy='median')
#imputer.strategy

num_pipe_parth = Pipeline([('imputer',SimpleImputer(missing_values=np.nan, strategy = "median")),
                         ('std_sclr',StandardScaler()),])

#x_train = num_pipe_parth.fit_transform(x_train)

#8.	Create a new Pipeline that has two steps the first is the num_pipe_firstname and the second is an SVM classifier with random state = last two digits of your student number. 
from sklearn.svm import SVC 

pipe_svm_parth = Pipeline([('pipe1',num_pipe_parth),
                           ('svc',SVC(random_state= 43))])






#10.	Define the grid search parameters in an object and name it param_grid, 

param_grid = [{
              'svc__kernel': ['linear', 'rbf','poly'],
              'svc__C':  [0.01,0.1, 1, 10, 100],
              'svc__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
              'svc__degree':[2,3]},]

#12.	Create a grid search object name it grid_search_firstname 


from sklearn.model_selection import GridSearchCV

grid_search_parth = GridSearchCV(estimator = pipe_svm_parth,
                                 param_grid = param_grid,
                                 scoring = 'accuracy',
                                 refit = True,
                                 verbose = 3)


#14.	Fit your training data to the gird search object

grid_search_parth.fit(x_train,y_train)

#15.	Print out the best parameters 

grid_search_parth.best_params_

#16.	Printout the best estimator and note it in your written response

grid_search_parth.best_estimator_
#17.	Fit the test data the grid search object and note it in your written response

y_pred = grid_search_parth.predict(x_test)
grid_search_parth.fit(x_test,y_test)


#18.	Printout the accuracy score and note it in your written response.
print(grid_search_parth.best_score_)

#19.	Create an object that holds the best model 
best_model_parth = grid_search_parth.best_estimator_

#20.	Save the model using the joblib  (dump).
import joblib
joblib.dump(best_model_parth, 
            'best_model_parth.pkl')

#21.	Save the full pipeline using the joblib – (dump).
joblib.dump(pipe_svm_parth, 
            'pipe_svm_parth.pkl')