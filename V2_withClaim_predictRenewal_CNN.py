# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 18:12:43 2018

@author: sakshij
"""

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.utils import np_utils

from tensorflow.keras.layers import Dense, Dropout,Flatten
from tensorflow.python.keras.layers.convolutional import Conv1D, MaxPooling2D

from tensorflow.keras.models import Sequential
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from tensorflow.keras.layers import Dropout, Reshape
from tensorflow.python.keras.layers.core import Activation

from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

#import yellowbrick
#from yellowbrick.classifier import DiscriminationThreshold
#from sklearn.linear_model import LogisticRegression

os.chdir("D:\\Projects\\Tableau_Train_Combine\\NN\\Codes")
from Plot_Metrics import plotMetricsWithThreshold

############################################################################
#Paths
model_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V2_withClaim_predictRenewal_63"
data_file_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Data"
data_file = "V2_WithClaim_RenewalPredict_Data_63"

############################################################################
#Report
os.chdir("D:\\Projects\\Tableau_Train_Combine\\NN\\Codes")
from Report_CNN import report

cols = ['Training Sample','Features','Sampling','Classifier','Confusion Matrix','_','Threshold','Classification Report','Precision','Recall','F1-score','Support','__','Accuracy','TPR','FPR','Misclassification_Rate','Specificity','Total_Precision','Model_Filename','Best_Parameters']
report_df = pd.DataFrame(columns=cols)

############################################################################
#Data        
os.chdir(data_file_path)

full_data = pd.read_csv(data_file + ".csv")

data_0 = full_data.copy(deep=True)
data_0_columns = data_0.columns.tolist()

scaler = MinMaxScaler(feature_range=(0,1))
data_0_scaled = scaler.fit_transform(data_0)

data_0 = pd.DataFrame(data_0_scaled, columns=data_0_columns)
############################################################################
#train and test split
y = data_0["Renewed"]
X = data_0.drop(["Renewed"],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)

X_train = X_train.values
X_test = X_test.values

y_train = y_train.values
y_test = y_test.values

X_train.shape
X_test.shape

#X_train_cnn = X_train.reshape(X_train.shape[0], -1).astype('float32')
#X_test_cnn = X_test.reshape(-1, X_test.shape[0] * X_test.shape[1], 1).astype('float32')

#X_train_cnn.shape
#X_test_cnn.shape

#X_train = X_train.reshape(X_train.shape[0], 28, 28 , 1).astype('float32')
#X_test = X_test.reshape(X_test.shape[0], 28, 28 , 1).astype('float32')

############################################################################
pd.value_counts(data_0.Renewed)
#1.0    426
#0.0    259
pd.value_counts(y_train)
#1.0    274
#0.0    171
pd.value_counts(y_test)
#1.0    152
#0.0     88
############################################################################
###Modelling
os.chdir(model_path)
os.getcwd()

##CNN 1
model_1 = Sequential()

#Convolutional Layers
model_1.add(Reshape((62, 1), input_shape=(62,)))
model_1.add(Conv1D(31, kernel_size=2, strides=2, padding="same", activation = 'relu'))

#Dense Layers
model_1.add(Flatten())
model_1.add(Dense(units=15, activation='relu'))
model_1.add(Dense(units=1, activation='sigmoid'))

model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model_1.fit(X_train, y_train, epochs=20, batch_size=20)

#Report Info
model_name = "CNN"
defined_params = ""
file_name = "CNN_1"
training_ratio = 0.65
num_layers = '1+1'
fig_name = file_name + "_metrics_plot"

#plot metrics by threshold
model_plot_info = plotMetricsWithThreshold(model_1, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#Report
threshold = 0.6

report_df = report(model_1, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 2
model_2 = Sequential()

#Convolutional Layers
model_2.add(Reshape((62, 1), input_shape=(62,)))
model_2.add(Conv1D(60, kernel_size=3, strides=1, padding="same", activation = 'relu'))
model_2.add(Conv1D(30, kernel_size=2, strides=2, padding="same", activation = 'relu'))

#Dense Layers
model_2.add(Flatten())
model_2.add(Dense(units=15, activation='relu'))
model_2.add(Dense(units=1, activation='sigmoid'))

model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model_2.fit(X_train, y_train, epochs=20, batch_size=20)

#Report Info
model_name = "CNN"
defined_params = ""
file_name = "CNN_2"
training_ratio = 0.65
num_layers = '2+1'
fig_name = file_name + "_metrics_plot"

#plot metrics by threshold
model_plot_info = plotMetricsWithThreshold(model_2, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#Report
threshold = 0.61

report_df = report(model_2, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 3
model_3 = Sequential()

#Convolutional Layers
model_3.add(Reshape((62, 1), input_shape=(62,)))
model_3.add(Conv1D(61, kernel_size=2, strides=1, padding="same", activation = 'relu'))
model_3.add(Conv1D(60, kernel_size=2, strides=1, padding="same", activation = 'relu'))
model_3.add(Conv1D(30, kernel_size=2, strides=2, padding="same", activation = 'relu'))

#Dense Layers
model_3.add(Flatten())
model_3.add(Dense(units=15, activation='relu'))
model_3.add(Dense(units=1, activation='sigmoid'))

model_3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model_3.fit(X_train, y_train, epochs=20, batch_size=20)

#Report Info
model_name = "CNN"
defined_params = ""
file_name = "CNN_3"
training_ratio = 0.65
num_layers = '3+1'
fig_name = file_name + "_metrics_plot"

#plot metrics by threshold
model_plot_info = plotMetricsWithThreshold(model_3, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#Report
threshold = 0.66

report_df = report(model_3, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 4
model_4 = Sequential()

#Convolutional Layers
model_4.add(Reshape((62, 1), input_shape=(62,)))
model_4.add(Conv1D(60, kernel_size=3, strides=1, padding="same", activation = 'relu'))
model_4.add(Conv1D(30, kernel_size=2, strides=2, padding="same", activation = 'relu'))

#Dense Layers
model_4.add(Flatten())
model_4.add(Dense(units=15, activation='relu'))
model_4.add(Dense(units=10, activation='relu'))
model_4.add(Dense(units=5, activation='relu'))
model_4.add(Dense(units=1, activation='sigmoid'))

model_4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model_4.fit(X_train, y_train, epochs=20, batch_size=20)

#Report Info
model_name = "CNN"
defined_params = ""
file_name = "CNN_4"
training_ratio = 0.65
num_layers = '2+3'
fig_name = file_name + "_metrics_plot"

#plot metrics by threshold
model_plot_info = plotMetricsWithThreshold(model_4, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#Report
threshold = 0.67

report_df = report(model_4, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 5
model_5 = Sequential()

#Convolutional Layers
model_5.add(Reshape((62, 1), input_shape=(62,)))
model_5.add(Conv1D(31, kernel_size=2, strides=2, padding="same", activation = 'relu'))
model_5.add(Conv1D(30, kernel_size=2, strides=1, padding="same", activation = 'relu'))

#Dense Layers
model_5.add(Flatten())
model_5.add(Dense(units=15, activation='relu'))
model_5.add(Dense(units=1, activation='sigmoid'))

model_5.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model_5.fit(X_train, y_train, epochs=20, batch_size=20)

#Report Info
model_name = "CNN"
defined_params = ""
file_name = "CNN_5"
training_ratio = 0.65
num_layers = '2+1'
fig_name = file_name + "_metrics_plot"

#plot metrics by threshold
model_plot_info = plotMetricsWithThreshold(model_5, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#Report
threshold = 0.57

report_df = report(model_5, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 6
model_6 = Sequential()

#Convolutional Layers
model_6.add(Reshape((62, 1), input_shape=(62,)))
model_6.add(Conv1D(61, kernel_size=2, strides=1, padding="same", activation = 'relu'))
model_6.add(Conv1D(60, kernel_size=2, strides=1, padding="same", activation = 'relu'))
model_6.add(Conv1D(30, kernel_size=2, strides=2, padding="same", activation = 'relu'))
model_6.add(Conv1D(10, kernel_size=3, strides=3, padding="same", activation = 'relu'))

#Dense Layers
model_6.add(Flatten())
model_6.add(Dense(units=15, activation='relu'))
model_6.add(Dense(units=1, activation='sigmoid'))

model_6.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model_6.fit(X_train, y_train, epochs=20, batch_size=20)

#Report Info
model_name = "CNN"
defined_params = ""
file_name = "CNN_6"
training_ratio = 0.65
num_layers = '4+1'
fig_name = file_name + "_metrics_plot"

#plot metrics by threshold
model_plot_info = plotMetricsWithThreshold(model_6, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#Report
threshold = 0.68

report_df = report(model_6, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 7
model_7 = Sequential()

#Convolutional Layers
model_7.add(Reshape((62, 1), input_shape=(62,)))
model_7.add(Conv1D(61, kernel_size=2, strides=1, padding="same", activation = 'relu'))
model_7.add(Conv1D(60, kernel_size=2, strides=1, padding="same", activation = 'relu'))
model_7.add(Conv1D(60, kernel_size=2, strides=1, padding="same", activation = 'relu'))
model_7.add(Conv1D(20, kernel_size=3, strides=3, padding="same", activation = 'relu'))

#Dense Layers
model_7.add(Flatten())
model_7.add(Dense(units=30, activation='relu'))
model_7.add(Dense(units=15, activation='relu'))
model_7.add(Dense(units=1, activation='sigmoid'))

model_7.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model_7.fit(X_train, y_train, epochs=20, batch_size=20)

#Report Info
model_name = "CNN"
defined_params = ""
file_name = "CNN_7"
training_ratio = 0.65
num_layers = '3+2'
fig_name = file_name + "_metrics_plot"

#plot metrics by threshold
model_plot_info = plotMetricsWithThreshold(model_7, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#Report
threshold = 0.47

report_df = report(model_7, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 8
model_8 = Sequential()

#Convolutional Layers
model_8.add(Reshape((62, 1), input_shape=(62,)))
model_8.add(Conv1D(61, kernel_size=2, strides=1, padding="same", activation = 'relu'))
model_8.add(Conv1D(60, kernel_size=2, strides=1, padding="same", activation = 'relu'))
model_8.add(Conv1D(59, kernel_size=2, strides=1, padding="same", activation = 'relu'))
model_8.add(Conv1D(40, kernel_size=20, strides=1, padding="same", activation = 'relu'))
model_8.add(Conv1D(20, kernel_size=2, strides=2, padding="same", activation = 'relu'))

#Dense Layers
model_8.add(Flatten())
model_8.add(Dense(units=30, activation='relu'))
model_8.add(Dense(units=1, activation='sigmoid'))

model_8.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model_8.fit(X_train, y_train, epochs=20, batch_size=20)

#Report Info
model_name = "CNN"
defined_params = ""
file_name = "CNN_8"
training_ratio = 0.65
num_layers = '5+1'
fig_name = file_name + "_metrics_plot"

#plot metrics by threshold
model_plot_info = plotMetricsWithThreshold(model_8, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#Report
threshold = 0.6

report_df = report(model_8, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

#report_df = report_df.drop(report_df.index[[21,22,23]])

##CNN 9
model_9 = Sequential()

#Convolutional Layers
model_9.add(Reshape((62, 1), input_shape=(62,)))
model_9.add(Conv1D(61, kernel_size=2, strides=1, padding="same", activation = 'relu'))
model_9.add(Conv1D(60, kernel_size=2, strides=1, padding="same", activation = 'relu'))
model_9.add(Conv1D(59, kernel_size=2, strides=1, padding="same", activation = 'relu'))
model_9.add(Conv1D(40, kernel_size=20, strides=1, padding="same", activation = 'relu'))
model_9.add(Conv1D(20, kernel_size=2, strides=2, padding="same", activation = 'relu'))

#Dense Layers
model_9.add(Flatten())
model_9.add(Dense(units=30, activation='relu'))
model_9.add(Dense(units=1, activation='sigmoid'))

model_9.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model_9.fit(X_train, y_train, epochs=20, batch_size=20)

#Report Info
model_name = "CNN"
defined_params = ""
file_name = "CNN_9"
training_ratio = 0.65
num_layers = '5+1'
fig_name = file_name + "_metrics_plot"

#plot metrics by threshold
model_plot_info = plotMetricsWithThreshold(model_9, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#Report
threshold = 0.74

report_df = report(model_9, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

#report_df = report_df.drop(report_df.index[[24,25,26]])

##CNN 10
model_10 = Sequential()

#Convolutional Layers
model_10.add(Reshape((62, 1), input_shape=(62,)))
model_10.add(Conv1D(31, kernel_size=2, strides=2, padding="same", activation = 'relu'))
model_10.add(Conv1D(20, kernel_size=11, strides=1, padding="same", activation = 'relu'))

#Dense Layers
model_10.add(Flatten())
model_10.add(Dense(units=30, activation='relu'))
model_10.add(Dense(units=1, activation='sigmoid'))

model_10.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the model
model_10.fit(X_train, y_train, epochs=20, batch_size=20)

#Report Info
model_name = "CNN"
defined_params = ""
file_name = "CNN_10"
training_ratio = 0.65
num_layers = '2+1'
fig_name = file_name + "_metrics_plot"

#plot metrics by threshold
model_plot_info = plotMetricsWithThreshold(model_10, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#Report
threshold = 0.64

report_df = report(model_10, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

os.getcwd()
report_df.to_csv("Report_CNN_V2_withClaim_predictRenewal_63cols_1.csv",index=False)