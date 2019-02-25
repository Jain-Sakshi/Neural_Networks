# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:13:04 2018

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

#import yellowbrick
#from yellowbrick.classifier import DiscriminationThreshold
#from sklearn.linear_model import LogisticRegression

os.chdir("D:\\Projects\\Tableau_Train_Combine\\NN\\Codes")
from Plot_Metrics import plotMetricsWithThreshold

def getThresholdRatio(y_pred,threshold):
    y_pred_with_threshold = []
    for prediction in y_pred:
        if prediction > threshold:
            y_pred_with_threshold.append(1)
        elif prediction < threshold:
            y_pred_with_threshold.append(0)
    
    vals = pd.value_counts(y_pred_with_threshold)
    print(vals[1]/vals[0])
    return vals[1]/vals[0]

############################################################################
#Paths
model_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V1_withClaim_predictRenewal"
data_file_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Data"
data_file = "V1_WithClaim_RenewalPredict_Data"

############################################################################
#Report
os.chdir("D:\\Projects\\Tableau_Train_Combine\\NN\\Codes")
from Report_CNN import report

cols = ['Training Sample','Features','Sampling','Classifier','Confusion Matrix','_','Threshold','Classification Report','Precision','Recall','F1-score','Support','__','Accuracy','TPR','FPR','Misclassification_Rate','Specificity','Total_Precision','Model_Filename','Best_Paramters']
report_df = pd.DataFrame(columns=cols)

report_df = pd.read_csv("Report_CNN_V1_withClaim_predictRenewal_1.csv")
#report_df.columns = cols

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
#0.0    296
#1.0    210
pd.value_counts(y_train)
#0.0    190
#1.0    138
pd.value_counts(y_test)
#0.0    106
#1.0     72
############################################################################
###Modelling
os.chdir(model_path)
os.getcwd()

##CNN 1
model_1 = Sequential()

#Convolutional Layers
model_1.add(Reshape((97, 1), input_shape=(97,)))
model_1.add(Conv1D(20, kernel_size=5, strides=5, padding="same", activation = 'relu'))

#Dense Layers
model_1.add(Flatten())
model_1.add(Dense(units=30, activation='relu'))
model_1.add(Dense(units=15, activation='relu'))
model_1.add(Dense(units=1, activation='sigmoid'))

model_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_1
model_1.fit(X_train, y_train, epochs=20, batch_size=20)

#predictions
y_pred = model_1.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))
y_pred = y_pred[0]

#Model Info to send for report()
model_name = "CNN"
defined_params = ""
file_name = "CNN_1"
training_ratio = 0.65
num_layers = '1+3'
fig_name = file_name + "_metrics_plot"

#plotting threshold vs metrics
model_1_plot_info = plotMetricsWithThreshold(y_test, y_pred, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#report()
threshold = 0.22

report_df = report(model_1, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

model_1_plot_info.to_csv("CNN_1_plot_info.csv",index=False)
report_df.to_csv("Report_CNN_V1_withClaim_predictRenewal_1.csv",index=False)

##CNN 2
model_2 = Sequential()

#Convolutional Layers
model_2.add(Reshape((97, 1), input_shape=(97,)))
model_2.add(Conv1D(48, kernel_size=3, strides=2, padding="same", activation = 'relu'))
model_2.add(Conv1D(20, kernel_size=2, strides=2, padding="same", activation = 'relu'))

#Dense Layers
model_2.add(Flatten())
model_2.add(Dense(units=30, activation='relu'))
model_2.add(Dense(units=15, activation='relu'))
model_2.add(Dense(units=1, activation='sigmoid'))

model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_2
model_2.fit(X_train, y_train, epochs=20, batch_size=20)

#Model Info to send for report()
model_name = "CNN"
defined_params = ""
file_name = "CNN_2"
training_ratio = 0.65
num_layers = '2+3'
fig_name = file_name + "_metrics_plot"

#plotting threshold vs metrics
model_2_plot_info = plotMetricsWithThreshold(model_2, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#report()
threshold = 0.46
report_df = report(model_2, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 3
model_3 = Sequential()

#Convolutional Layers
model_3.add(Reshape((97, 1), input_shape=(97,)))
model_3.add(Conv1D(48, kernel_size=2, strides=2, padding="same", activation = 'relu'))
model_3.add(Conv1D(40, kernel_size=8, strides=1, padding="same", activation = 'relu'))
model_3.add(Conv1D(20, kernel_size=2, strides=2, padding="same", activation = 'relu'))

#Dense Layers
model_3.add(Flatten())
model_3.add(Dense(units=30, activation='relu'))
model_3.add(Dense(units=15, activation='relu'))
model_3.add(Dense(units=1, activation='sigmoid'))

model_3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_3
model_3.fit(X_train, y_train, epochs=20, batch_size=20)

#Model Info to send for report()
model_name = "CNN"
defined_params = ""
file_name = "CNN_3"
training_ratio = 0.65
num_layers = '3+3'
fig_name = file_name + "_metrics_plot"

#plotting threshold vs metrics
model_3_plot_info = plotMetricsWithThreshold(model_3, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#report()
threshold = 0.3

report_df = report(model_3, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 4
model_4 = Sequential()

#Convolutional Layers
model_4.add(Reshape((97, 1), input_shape=(97,)))
model_4.add(Conv1D(25, kernel_size=4, strides=4, padding="same", activation = 'relu'))

#Dense Layers
model_4.add(Flatten())
model_4.add(Dense(units=30, activation='relu'))
model_4.add(Dense(units=15, activation='relu'))
model_4.add(Dense(units=1, activation='sigmoid'))

model_4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_4
model_4.fit(X_train, y_train, epochs=20, batch_size=20)

#Model Info to send for report()
model_name = "CNN"
defined_params = ""
file_name = "CNN_4"
training_ratio = 0.65
num_layers = '1+3'
fig_name = file_name + "_metrics_plot"

os.getcwd()
#plotting threshold vs metrics
model_4_plot_info = plotMetricsWithThreshold(model_4, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#report()
threshold = 0.2

report_df = report(model_4, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 5
model_5 = Sequential()

#Convolutional Layers
model_5.add(Reshape((97, 1), input_shape=(97,)))
model_5.add(Conv1D(33, kernel_size=3, strides=3, padding="same", activation = 'relu'))
model_5.add(Conv1D(20, kernel_size=14, strides=1, padding="same", activation = 'relu'))

#Dense Layers
model_5.add(Flatten())
model_5.add(Dense(units=30, activation='relu'))
model_5.add(Dense(units=15, activation='relu'))
model_5.add(Dense(units=1, activation='sigmoid'))

model_5.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_5
model_5.fit(X_train, y_train, epochs=20, batch_size=20)

#Model Info to send for report()
model_name = "CNN"
defined_params = ""
file_name = "CNN_5"
training_ratio = 0.65
num_layers = '2+3'
fig_name = file_name + "_metrics_plot"

os.getcwd()
#plotting threshold vs metrics
model_5_plot_info = plotMetricsWithThreshold(model_5, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#report()
threshold = 0.22

report_df = report(model_5, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 6
model_6 = Sequential()

#Convolutional Layers
model_6.add(Reshape((97, 1), input_shape=(97,)))
model_6.add(Conv1D(48, kernel_size=2, strides=2, padding="same", activation = 'relu'))
model_6.add(Conv1D(40, kernel_size=8, strides=1, padding="same", activation = 'relu'))
model_6.add(Conv1D(20, kernel_size=2, strides=2, padding="same", activation = 'relu'))
model_6.add(Conv1D(10, kernel_size=2, strides=2, padding="same", activation = 'relu'))

#Dense Layers
model_6.add(Flatten())
model_6.add(Dense(units=30, activation='relu'))
model_6.add(Dense(units=15, activation='relu'))
model_6.add(Dense(units=1, activation='sigmoid'))

model_6.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_6
model_6.fit(X_train, y_train, epochs=20, batch_size=20)

#Model Info to send for report()
model_name = "CNN"
defined_params = ""
file_name = "CNN_6"
training_ratio = 0.65
num_layers = '4+3'
fig_name = file_name + "_metrics_plot"

os.getcwd()
#plotting threshold vs metrics
model_6_plot_info = plotMetricsWithThreshold(model_6, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#report()
threshold = 0.33

report_df = report(model_6, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

threshold = 0.42

report_df = report(model_6, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

#CNN 3_1
old_model_filename="CNN_3"
json_file = open(old_model_filename+ '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_3_loaded = model_from_json(loaded_model_json)
model_3_loaded.load_weights(old_model_filename + ".h5")

model_3_loaded.summary()

#Create dropout layers
dropout1 = Dropout(0.4)
dropout2 = Dropout(0.4)

#store dense layers separately after which dropout
#layers have to be added 
dense_layer_1 = model_3_loaded.layers[-3]
dense_layer_2 = model_3_loaded.layers[-2]
dense_layer_3 = model_3_loaded.layers[-1]

model_3_loaded.layers[-3].output

#Reconnect the layers
x = dropout1(dense_layer_1.output)
x = dense_layer_2(x)
x = dropout2(x)

model_3_loaded.summary()

predictors = dense_layer_3(x)

#Create the new model
model_3_1 = Model(inputs = model_3_loaded.input, outputs = predictors)

#Fit the model_3_1
model_3_1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_3_1.fit(X_train, y_train, epochs=20, batch_size=20)

#Model Info to send for report()
model_name = "CNN"
defined_params = "dropout"
file_name = "CNN_3_1"
training_ratio = 0.65
num_layers = '4+3'
fig_name = file_name + "_metrics_plot"

os.getcwd()
#plotting threshold vs metrics
model_3_1_plot_info = plotMetricsWithThreshold(model_3_1, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#report()
threshold = 0.38

report_df = report(model_3_1, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')


#CNN 3_2
old_model_filename="CNN_3"
json_file = open(old_model_filename+ '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_3_loaded = model_from_json(loaded_model_json)
model_3_loaded.load_weights(old_model_filename + ".h5")

model_3_loaded.summary()

#Create dropout layers
dropout1 = Dropout(0.4)
dropout2 = Dropout(0.4)

#Create the new model
model_3_2 = model_3_loaded

#Fit the model_3_2
model_3_2.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model_3_2.fit(X_train, y_train, epochs=20, batch_size=20)

#Model Info to send for report()
model_name = "CNN"
defined_params = "optimizer - RMSprop"
file_name = "CNN_3_2"
training_ratio = 0.65
num_layers = '4+3'
fig_name = file_name + "_metrics_plot"

os.getcwd()
#plotting threshold vs metrics
model_3_2_plot_info = plotMetricsWithThreshold(model_3_2, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)

#report()
threshold = 0.31

report_df = report(model_3_2, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

report_df.to_csv("Report_CNN_V1_withClaim_predictRenewal_1.csv",index=False)