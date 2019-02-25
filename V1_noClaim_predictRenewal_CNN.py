# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:16:55 2018

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
model_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V1_noClaim_predictRenewal"
data_file_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V1_noClaim_predictRenewal"
data_file = "V1_NoClaim_RenewalPredict_Data"

############################################################################
#Report
os.chdir("D:\\Projects\\Tableau_Train_Combine\\NN\\Codes")
from Report_CNN import report

cols = ['Training Sample','Features','Sampling','Classifier','Confusion Matrix','_','Threshold','Classification Report','Precision','Recall','F1-score','Support','__','Accuracy','TPR','FPR','Misclassification_Rate','Specificity','Total_Precision','Model_Filename','Best_Paramters']
report_df = pd.DataFrame(columns=cols)
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
#0.0    2257
#1.0     742
pd.value_counts(y_train)
#0.0    1463
#1.0     486
pd.value_counts(y_test)
#0.0    794
#1.0    256
############################################################################
###Modelling
os.chdir(model_path)
os.getcwd()
##CNN 1
model = Sequential()

#Convolutional Layers
model.add(Reshape((55, 1)))
model.add(Conv1D(50, kernel_size=5, strides=1, padding="same", activation = 'relu'))

#Dense Layers
model.add(Flatten())
model.add(Dense(units=60, input_dim=55, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(units=30, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(units=15, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model
y_pred = model.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.37)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_1"
training_ratio = 0.65
num_layers = '1+4'
threshold = 0.37

report_df = report(model, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

#Yellowbrick Threshold Classifier
model_test = LogisticRegression()
visualizer = DiscriminationThreshold(model_test)
temp_a = visualizer.fit(X_train, y_train,solver='lbfgs')
temp_a.draw()
temp_a.finalize()
temp_a.poof()
visualizer.poof()

##CNN 2
model_2 = Sequential()

#Convolutional Layers
model_2.add(Reshape((55, 1)))
model_2.add(Conv1D(50, kernel_size=5, strides=1, padding="same", activation = 'relu'))
#model_2.add(Conv1D(10, kernel_size=5, strides=5, padding="same", activation = 'relu'))

#Dense Layers
model_2.add(Flatten())
model_2.add(Dense(units=60, input_dim=55, activation='relu'))
model_2.add(Dropout(0.1))
model_2.add(Dense(units=30, activation='relu'))
model_2.add(Dropout(0.1))
model_2.add(Dense(units=15, activation='relu'))
model_2.add(Dropout(0.3))
model_2.add(Dense(units=1, activation='sigmoid'))

model_2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_2
model_2.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_2
y_pred = model_2.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.3)

#Report
model_name = "CNN"
defined_params = "CNN1 + dropout"
file_name = "CNN_2"
training_ratio = 0.65
num_layers = '1+4'
threshold = 0.3

report_df = report(model_2, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 3
model_3 = Sequential()

#Convolutional Layers
model_3.add(Reshape((55, 1)))
model_3.add(Conv1D(50, kernel_size=5, strides=1, padding="same", activation = 'relu'))
model_3.add(Conv1D(10, kernel_size=5, strides=5, padding="same", activation = 'relu'))

#Dense Layers
model_3.add(Flatten())
model_3.add(Dense(units=5, activation='relu'))
#model_3.add(Dropout(0.1))
model_3.add(Dense(units=1, activation='sigmoid'))

model_3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_3
model_3.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_3
y_pred = model_3.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.27)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_3"
training_ratio = 0.65
num_layers = '2+2'
threshold = 0.27

report_df = report(model_3, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 4
model_4 = Sequential()

#Convolutional Layers
model_4.add(Reshape((55, 1)))
model_4.add(Conv1D(50, kernel_size=5, strides=1, padding="same", activation = 'relu'))
model_4.add(Conv1D(24, kernel_size=4, strides=5, padding="same", activation = 'relu'))

#Dense Layers
model_4.add(Flatten())
model_4.add(Dense(units=30, activation='relu'))
model_4.add(Dense(units=15, activation='relu'))
#model_4.add(Dropout(0.1))
model_4.add(Dense(units=1, activation='sigmoid'))

model_4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_4
model_4.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_4
y_pred = model_4.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.35)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_4"
training_ratio = 0.65
num_layers = '2+2'
threshold = 0.35

report_df = report(model_4, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 5
model_5 = Sequential()

#Convolutional Layers
model_5.add(Reshape((55, 1)))
model_5.add(Conv1D(50, kernel_size=5, strides=1, padding="same", activation = 'relu'))
model_5.add(Conv1D(24, kernel_size=4, strides=5, padding="same", activation = 'relu'))

#Dense Layers
model_5.add(Flatten())
model_5.add(Dense(units=30, activation='relu'))
model_5.add(Dense(units=15, activation='relu'))
model_5.add(Dense(units=7, activation='relu'))
#model_5.add(Dropout(0.1))
model_5.add(Dense(units=1, activation='sigmoid'))

model_5.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_5
model_5.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_5
y_pred = model_5.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.35)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_5"
training_ratio = 0.65
num_layers = '2+4'
threshold = 0.35

report_df = report(model_5, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 6
model_6 = Sequential()

#Convolutional Layers
model_6.add(Reshape((55, 1)))
model_6.add(Conv1D(11, kernel_size=5, strides=5, padding="same", activation = 'relu'))

#Dense Layers
model_6.add(Flatten())
model_6.add(Dense(units=20, activation='relu'))
model_6.add(Dense(units=10, activation='relu'))
#model_6.add(Dropout(0.1))
model_6.add(Dense(units=1, activation='sigmoid'))

model_6.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_6
model_6.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_6
y_pred = model_6.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.22)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_6"
training_ratio = 0.65
num_layers = '1+3'
threshold = 0.22

report_df = report(model_6, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 7
model_7 = Sequential()

#Convolutional Layers
model_7.add(Reshape((55, 1)))
model_7.add(Conv1D(11, kernel_size=5, strides=5, padding="same", activation = 'relu'))

#Dense Layers
model_7.add(Flatten())
model_7.add(Dense(units=10, activation='relu'))
model_7.add(Dense(units=5, activation='relu'))
#model_7.add(Dropout(0.1))
model_7.add(Dense(units=1, activation='sigmoid'))

model_7.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_7
model_7.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_7
y_pred = model_7.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.315)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_7"
training_ratio = 0.65
num_layers = '1+3'
threshold = 0.315

report_df = report(model_7, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 8
model_8 = Sequential()

#Convolutional Layers
model_8.add(Reshape((55, 1)))
model_8.add(Conv1D(50, kernel_size=5, strides=1, padding="same", activation = 'relu'))
model_8.add(Conv1D(24, kernel_size=5, strides=2, padding="same", activation = 'relu'))

#Dense Layers
model_8.add(Flatten())
model_8.add(Dense(units=10, activation='relu'))
model_8.add(Dense(units=5, activation='relu'))
#model_8.add(Dropout(0.1))
model_8.add(Dense(units=1, activation='sigmoid'))

model_8.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_8
model_8.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_8
y_pred = model_8.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.35)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_8"
training_ratio = 0.65
num_layers = '2+3'
threshold = 0.35

report_df = report(model_8, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 9
model_9 = Sequential()

#Convolutional Layers
model_9.add(Reshape((55, 1)))
model_9.add(Conv1D(50, kernel_size=5, strides=1, padding="same", activation = 'relu'))
model_9.add(Conv1D(24, kernel_size=5, strides=2, padding="same", activation = 'relu'))

#Dense Layers
model_9.add(Flatten())
model_9.add(Dense(units=30, activation='relu'))
model_9.add(Dense(units=15, activation='relu'))
#model_9.add(Dropout(0.1))
model_9.add(Dense(units=1, activation='sigmoid'))

model_9.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_9
model_9.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_9
y_pred = model_9.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.31)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_9"
training_ratio = 0.65
num_layers = '2+3'
threshold = 0.31

report_df = report(model_9, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 10
model_10 = Sequential()

#Convolutional Layers
model_10.add(Reshape((55, 1)))
model_10.add(Conv1D(50, kernel_size=5, strides=1, padding="same", activation = 'relu'))
model_10.add(Conv1D(25, kernel_size=2, strides=2, padding="same", activation = 'relu'))

#Dense Layers
model_10.add(Flatten())
model_10.add(Dense(units=30, activation='relu'))
model_10.add(Dense(units=15, activation='relu'))
#model_10.add(Dropout(0.1))
model_10.add(Dense(units=1, activation='sigmoid'))

model_10.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_10
model_10.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_10
y_pred = model_10.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.42)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_10"
training_ratio = 0.65
num_layers = '2+3'
threshold = 0.42

report_df = report(model_10, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 11
model_11 = Sequential()

#Convolutional Layers
#model_11.add(Reshape((55, 1)))
model_11.add(Reshape((55, 1), input_shape=(55,)))
model_11.add(Conv1D(50, kernel_size=5, strides=1, padding="same", activation = 'relu'))
model_11.add(Conv1D(24, kernel_size=4, strides=5, padding="same", activation = 'relu'))
model_11.add(Conv1D(23, kernel_size=2, strides=1, padding="same", activation = 'relu'))

#Dense Layers
model_11.add(Flatten())
model_11.add(Dense(units=30, activation='relu'))
model_11.add(Dense(units=15, activation='relu'))
#model_11.add(Dropout(0.1))
model_11.add(Dense(units=1, activation='sigmoid'))

model_11.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_11
model_11.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_11
y_pred = model_11.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.3)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_11_1"
training_ratio = 0.65
num_layers = '3+3'
#threshold = 0.27
threshold = 0.3

report_df = report(model_11, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 12
model_12 = Sequential()

#Convolutional Layers
model_12.add(Reshape((55, 1)))
model_12.add(Conv1D(50, kernel_size=5, strides=1, padding="same", activation = 'relu'))
model_12.add(Conv1D(24, kernel_size=5, strides=4, padding="same", activation = 'relu'))

#Dense Layers
model_12.add(Flatten())
model_12.add(Dense(units=30, activation='relu'))
model_12.add(Dense(units=15, activation='relu'))
#model_12.add(Dropout(0.1))
model_12.add(Dense(units=1, activation='sigmoid'))

model_12.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_12
model_12.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_12
y_pred = model_12.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.33)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_12"
training_ratio = 0.65
num_layers = '3+3'
threshold = 0.33

report_df = report(model_12, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 13
model_13 = Sequential()

#Convolutional Layers
model_13.add(Reshape((55, 1)))
model_13.add(Conv1D(50, kernel_size=5, strides=1, padding="same", activation = 'relu'))
model_13.add(Conv1D(24, kernel_size=5, strides=4, padding="same", activation = 'relu'))
model_13.add(Conv1D(6, kernel_size=4, strides=4, padding="same", activation = 'relu'))

#Dense Layers
model_13.add(Flatten())
model_13.add(Dense(units=30, activation='relu'))
model_13.add(Dense(units=15, activation='relu'))
#model_13.add(Dropout(0.1))
model_13.add(Dense(units=1, activation='sigmoid'))

model_13.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_13
model_13.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_13
y_pred = model_13.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.28)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_13"
training_ratio = 0.65
num_layers = '3+3'
threshold = 0.28

report_df = report(model_13, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 14
model_14 = Sequential()

#Convolutional Layers
model_14.add(Reshape((55, 1)))
model_14.add(Conv1D(50, kernel_size=5, strides=1, padding="same", activation = 'relu'))
model_14.add(Conv1D(24, kernel_size=5, strides=4, padding="same", activation = 'relu'))
model_14.add(Conv1D(6, kernel_size=4, strides=4, padding="same", activation = 'relu'))

#Dense Layers
model_14.add(Flatten())
#model_14.add(Dropout(0.1))
model_14.add(Dense(units=1, activation='sigmoid'))

model_14.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_14
model_14.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_14
y_pred = model_14.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.3)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_14"
training_ratio = 0.65
num_layers = '3+1'
threshold = 0.3

report_df = report(model_14, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')
##CNN 15
model_15 = Sequential()

#Convolutional Layers
model_15.add(Reshape((55, 1)))
model_15.add(Conv1D(50, kernel_size=5, strides=1, padding="same", activation = 'relu'))
model_15.add(Conv1D(24, kernel_size=4, strides=5, padding="same", activation = 'relu'))
model_15.add(Conv1D(12, kernel_size=3, strides=4, padding="same", activation = 'relu'))

#Dense Layers
model_15.add(Flatten())
model_15.add(Dense(units=30, activation='relu'))
model_15.add(Dense(units=15, activation='relu'))
#model_15.add(Dropout(0.1))
model_15.add(Dense(units=1, activation='sigmoid'))

model_15.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_15
model_15.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_15
y_pred = model_15.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.35)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_15"
training_ratio = 0.65
num_layers = '3+3'
threshold = 0.35

report_df = report(model_15, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 16
model_16 = Sequential()

#Convolutional Layers
model_16.add(Reshape((55, 1)))
model_16.add(Conv1D(50, kernel_size=5, strides=1, padding="same", activation = 'relu'))
model_16.add(Conv1D(24, kernel_size=4, strides=5, padding="same", activation = 'relu'))
model_16.add(Conv1D(23, kernel_size=2, strides=1, padding="same", activation = 'relu'))
model_16.add(Conv1D(6, kernel_size=3, strides=4, padding="same", activation = 'relu'))

#Dense Layers
model_16.add(Flatten())
model_16.add(Dense(units=30, activation='relu'))
model_16.add(Dense(units=15, activation='relu'))
#model_16.add(Dropout(0.1))
model_16.add(Dense(units=1, activation='sigmoid'))

model_16.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_16
model_16.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_16
y_pred = model_16.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.4)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_16"
training_ratio = 0.65
num_layers = '4+3'
threshold = 0.4

report_df = report(model_16, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 17
model_17 = Sequential()

#Convolutional Layers
model_17.add(Reshape((55, 1)))
model_17.add(Conv1D(50, kernel_size=5, strides=1, padding="same", activation = 'relu'))
model_17.add(Conv1D(24, kernel_size=4, strides=5, padding="same", activation = 'relu'))
model_17.add(Conv1D(23, kernel_size=2, strides=1, padding="same", activation = 'relu'))
model_17.add(Conv1D(6, kernel_size=3, strides=3, padding="same", activation = 'relu'))

#Dense Layers
model_17.add(Flatten())
model_17.add(Dense(units=30, activation='relu'))
model_17.add(Dense(units=15, activation='relu'))
#model_17.add(Dropout(0.1))
model_17.add(Dense(units=1, activation='sigmoid'))

model_17.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_17
model_17.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_17
y_pred = model_17.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.33)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_17"
training_ratio = 0.65
num_layers = '4+3'
threshold = 0.33

report_df = report(model_17, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 18
model_18 = Sequential()

#Convolutional Layers
model_18.add(Reshape((55, 1)))
model_18.add(Conv1D(50, kernel_size=5, strides=1, padding="same", activation = 'relu'))
model_18.add(Conv1D(24, kernel_size=4, strides=5, padding="same", activation = 'relu'))
model_18.add(Conv1D(23, kernel_size=2, strides=1, padding="same", activation = 'relu'))

#Dense Layers
model_18.add(Flatten())
model_18.add(Dense(units=30, activation='relu'))
model_18.add(Dense(units=15, activation='relu'))
model_18.add(Dense(units=7, activation='relu'))
#model_18.add(Dropout(0.1))
model_18.add(Dense(units=1, activation='sigmoid'))

model_18.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_18
model_18.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_18
y_pred = model_18.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.35)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_18"
training_ratio = 0.65
num_layers = '3+4'
threshold = 0.35

report_df = report(model_18, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

##CNN 19
model_19 = Sequential()

#Convolutional Layers
model_19.add(Reshape((55, 1)))
model_19.add(Conv1D(50, kernel_size=5, strides=1, padding="same", activation = 'relu'))
model_19.add(Conv1D(24, kernel_size=4, strides=5, padding="same", activation = 'relu'))
model_19.add(Conv1D(23, kernel_size=2, strides=2, padding="same", activation = 'relu'))

#Dense Layers
model_19.add(Flatten())
model_19.add(Dense(units=30, activation='relu'))
model_19.add(Dense(units=15, activation='relu'))
#model_19.add(Dense(units=7, activation='relu'))
#model_19.add(Dropout(0.1))
model_19.add(Dense(units=1, activation='sigmoid'))

model_19.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model_19
model_19.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_19
y_pred = model_19.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.46)

#Report
model_name = "CNN"
defined_params = ""
file_name = "CNN_19"
training_ratio = 0.65
num_layers = '3+3'
threshold = 0.46

report_df = report(model_19, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

report_df.to_csv("Report_CNN_V1_noClaim_predictRenewal.csv",index=False)

#model_19.layers[1].output


#Testing for creating result graph

from sklearn.metrics import precision_score, recall_score
all_precisions = []
all_recalls = []
threshold_range = np.arange(0,1,0.01)
for threshold in threshold_range:
    y_pred_threshold = (y_pred[0] > threshold).astype('int')
    all_precisions.append(precision_score(y_test, y_pred_threshold))
    all_recalls.append(recall_score(y_test, y_pred_threshold))

import matplotlib.pyplot as plt
plt.plot(threshold_range,all_precisions,threshold_range,all_recalls)
plot_cols = ["precision","recall"] #
plt.legend(cols)
plt.show()

t = threshold_range
plt.plot(t, all_precisions, 'r--', t, all_recalls, 'g^')
plt.show()

plt.savefig("Temp.png")

plot_df = pd.DataFrame()
plot_df["Total_Precision"] = all_precisions
plot_df["Total_Recall"] = all_recalls

import matplotlib
import pandas as pd

import matplotlib.pyplot as plt

plot_df.plot()
plot_df.index = threshold_range

fig, ax = plt.subplots(figsize=(5, 3))
ax.stackplot(threshold_range, [all_precisions, all_recalls], labels=cols)
ax.set_title('Metrics changing with Threshold')
ax.legend(loc='upper left')
ax.set_ylabel('Metric')
ax.set_xlabel('Threshold')
#ax.set_xlim(xmin=yrs[0], xmax=yrs[-1])
fig.tight_layout()
fig.plot()

os.getcwd()

if all_precisions:
    plt.plot(threshold_range, all_precisions)
if all_recalls:
    plt.plot([threshold_range, all_recalls, all_precisions])
plt.title('Metrics changing with Threshold')
plt.legend(plot_cols)
plt.ylabel('Metric')
plt.xlabel('Threshold')
#plt.show()
plt.savefig("Visualizing_a_complicated_chart.png")
##CNN 20
#Load the pre-existing model
json_file_11 = open('CNN_11_1.json', 'r')
loaded_model_json = json_file_11.read()
json_file_11.close()
model_11_loaded = model_from_json(loaded_model_json)
model_11_loaded.load_weights("CNN_11_1.h5")

#Create dropout layers
dropout1 = Dropout(0.4)
dropout2 = Dropout(0.4)

#store dense layers separately after which dropout
#layers have to be added 
len(model_11_loaded.layers)
dense_layer_1 = model_11_loaded.layers[-3]
dense_layer_2 = model_11_loaded.layers[-2]
dense_layer_3 = model_11_loaded.layers[-1]

model_11_loaded.layers[-3].output

#Reconnect the layers
x = dropout1(dense_layer_1.output)
x = dense_layer_2(x)
x = dropout2(x)

model_11_loaded.summary()

predictors = dense_layer_3(x)

#Create the new model

model_20 = Model(inputs = model_11_loaded.input, outputs = predictors)

# =============================================================================
# #testing plotMetricsWithThreshold
# os.getcwd()
# import Plot_Metrics
# from Plot_Metrics import plotMetricsWithThreshold
# plotMetricsWithThreshold(y_test, y_pred, fig_name="test", fig_path = None,
#                              precision="majority", recall="minority", 
#                              fpoint5score = True, f1score=True, f2score=True, fbetascore=0.5,
#                              TPR=True, FPR=True, specificity=True,
#                              accuracy=True, misclassification=True,
#                              display_info=True)
# 
# =============================================================================
# Fit the model_20
model_20.fit(X_train, y_train, epochs=20, batch_size=20)
# evaluate the model_20
y_pred = model_20.predict(X_test)
y_pred = y_pred.reshape(1,len(y_pred))

getThresholdRatio(y_pred[0],0.3)

#Report
model_name = "CNN"
defined_params = "Adding dropout to CNN_11"
file_name = "none"
training_ratio = 0.65
num_layers = '3+3'
threshold = 0.46

report_df = report(model_20, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic')

#Making threshold diagrams for every model
cnn_report = pd.read_csv("Report_CNN_V1_noClaim_predictRenewal.csv")

model_plot_info = []
filenames = []
for filename in cnn_report.Model_Filename:
    if type(filename) == str:
        if len(filename) > 2:
            print(filename)
            filenames.append(filename)   
            
index = 0

#run 'no of models' times
filename = filenames[index]
print(filename)
json_file = open(filename + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(filename + ".h5")

y_pred = model.predict(X_test)

fig_name = filename + "_plot"
plot_info_one = plotMetricsWithThreshold(model, X_test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True)
model_plot_info.append(plot_info_one)
index+=1