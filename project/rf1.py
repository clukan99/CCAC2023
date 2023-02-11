### This is a simple model with one middle layer for a logistic regression neural network

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import sklearn
from datetime import date
from datetime import datetime
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import accuracy_score
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


random.seed(1234)

print("Reading in the data")
########
now = datetime.now()
print("now =", now)
########
trucks = pd.read_csv("../fully_cleaned_super_nice_data.csv")
trucks = trucks.dropna()
print("Data reading is complete")
########
now = datetime.now()
print("now =", now)
########

y_data = trucks[['Sale_Price']]
X_data = trucks.drop(labels= ['Sale_Price'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, train_size= 0.8, random_state= 440)

X_train_columns = X_train.columns
X_test_columns = X_test.columns
y_train_columns = y_train.columns
y_test_columns = y_test.columns

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
########
now = datetime.now()
print("now =", now)
########
##################################

##################################


##### Defining the model#####
#rf_model = RandomForestClassifier(n_estimators=50, max_features="auto", random_state=1234)
###############################

#### Grid search ############
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
############################

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)

print(y_train)
################################
rf = RandomForestRegressor(n_estimators= 400, min_samples_split= 2, min_samples_leaf= 1, max_features= 'sqrt', max_depth= 110, bootstrap= False, random_state= 1234, verbose= 2, n_jobs= 32)
#########Random search training#############
#print("Starting random search!!!!!")
print("Searching...")
#rf_random = RandomizedSearchCV(estimator = rf_model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=1234)
############################################
print("Fitting.....")
#rf_random.fit(X_train_np, y_train_np.ravel())
#######Viewing the best parameters#####
print("Viewing the best parameters")
#rf_random.best_params_
######################################
rf.fit(X_train, y_train.ravel())

y_pred = rf.predict(X_test)


dicked = {'prediction': list(y_pred),
          'test': list(y_test)}

import IPython; IPython.embed(); exit(1)

undicked = pd.DataFrame(dicked)
undicked.to_csv("../results/RF_model.csv")

#######Modelling#####