# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 15:08:29 2019

@author: sakshij
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import preprocessing 
from sklearn.metrics import mean_squared_error 
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from tpot import TPOTClassifier

#Reading the data
os.chdir("D:\\Datasets\\Forest_Cover_Type")
data_df = pd.read_csv("covtype.csv", index_col = 0)

data_df.dtypes

#encoder = LabelEncoder()
#target = encoder.fit_transform(data_df["Species"])
#data_df["Species"] = target
#
#y = data_df["Species"]
#X = data_df.drop(["Species"], axis=1)

y = data_df["Cover_Type"]
X = data_df.drop(["Cover_Type"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

tpot = TPOTClassifier(generations=5, population_size=20, random_state = 0, verbosity = 3)
tpot.fit(X_train, y_train)

tpot.export("Forest_Cover_pipeline_1.py")

y_pred = tpot.predict(X_test)

accuracy_score(y_test, y_pred)

# =============================================================================
# Results (IRIS Dataset):
#    {
#     'generations' : 10,
#     'population' : 200
#     }
#    accuracy : 0.9333333333333333
#
#    {
#     'generations' : 30,
#     'population' : 200
#     }
#    accuracy : 1.0
#       
# =============================================================================

Results (Forest Cover Dataset):
    {
     generations=5, 
     population_size=20, 
     max_eval_time_mins = 1, 
     random_state = 0, 
     verbosity = 3
     }
    accuracy : 0.6562901597209473