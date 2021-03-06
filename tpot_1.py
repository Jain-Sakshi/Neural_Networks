# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:11:30 2019

@author: sakshij
"""

#import basic libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
#%matplotlib inline 
from sklearn import preprocessing 
from sklearn.metrics import mean_squared_error 
import os

#Reading the data
os.chdir("D:\\Datasets\\Big_Mart_Sales_III")
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")

## preprocessing 
### mean imputations 
train['Item_Weight'].fillna((train['Item_Weight'].mean()), inplace=True)
test['Item_Weight'].fillna((test['Item_Weight'].mean()), inplace=True)
### reducing fat content to only two categories 
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace(['low fat','LF'], ['Low Fat','Low Fat']) 
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace(['reg'], ['Regular']) 
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace(['low fat','LF'], ['Low Fat','Low Fat']) 
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace(['reg'], ['Regular']) 
train['Outlet_Establishment_Year'] = 2013 - train['Outlet_Establishment_Year'] 
test['Outlet_Establishment_Year'] = 2013 - test['Outlet_Establishment_Year'] 

train['Outlet_Size'].fillna('Small',inplace=True)
test['Outlet_Size'].fillna('Small',inplace=True)

train['Item_Visibility'] = np.sqrt(train['Item_Visibility'])
test['Item_Visibility'] = np.sqrt(test['Item_Visibility'])

col = ['Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Fat_Content']
test['Item_Outlet_Sales'] = 0
combi = train.append(test)
number = preprocessing.LabelEncoder()
for i in col:
 combi[i] = number.fit_transform(combi[i].astype('str'))
 combi[i] = combi[i].astype('object')
train = combi[:train.shape[0]]
test = combi[train.shape[0]:]
test.drop('Item_Outlet_Sales',axis=1,inplace=True)

## removing id variables 
tpot_train = train.drop(['Outlet_Identifier','Item_Type','Item_Identifier'],axis=1)
tpot_test = test.drop(['Outlet_Identifier','Item_Type','Item_Identifier'],axis=1)
target = tpot_train['Item_Outlet_Sales']
tpot_train.drop('Item_Outlet_Sales',axis=1,inplace=True)

#converting object data type columns to float
tpot_train['Item_Fat_Content'] = tpot_train['Item_Fat_Content'].apply(lambda x : float(x))
tpot_train['Outlet_Size'] = tpot_train['Outlet_Size'].apply(lambda x : float(x))
tpot_train['Outlet_Location_Type'] = tpot_train['Outlet_Location_Type'].apply(lambda x : float(x))
tpot_train['Outlet_Type'] = tpot_train['Outlet_Type'].apply(lambda x : float(x))

# finally building model using tpot library
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tpot_train, target,
 train_size=0.75, test_size=0.25)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_boston_pipeline.py')