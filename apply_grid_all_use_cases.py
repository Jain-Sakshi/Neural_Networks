# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 18:18:29 2018

@author: sakshij
"""

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils import np_utils

from tensorflow.python.keras.layers import Dense, Dropout,Flatten
from tensorflow.python.keras.layers.convolutional import Conv1D, MaxPooling2D

from tensorflow.python.keras.models import Sequential
import pandas as pd
import os

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from tensorflow.python.keras.layers import Dropout, Reshape
from tensorflow.python.keras.layers.core import Activation

from tensorflow.python.keras.models import model_from_json, Model
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.python.keras.callbacks import LearningRateScheduler

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tensorflow.python import keras
from tensorflow.python.keras.models import model_from_json

import math
import pickle

os.chdir("D:\\Projects\\Tableau_Train_Combine\\NN\\Codes")
from Plot_Metrics import plotMetricsWithThreshold
from Report_CNN import saveModelandModelInfo
from Filter_Thresholds import combineModelReports, filterThresholds
############################################################################
def getUseCaseDataInfo(use_case):
    if use_case == "V1_noClaim":
        original_models_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V1_noClaim_predictRenewal"
        model_path_pretrained = original_models_path + "\\grid"
        data_file = "V1_NoClaim_RenewalPredict_Data.csv"
        no_of_models = 6
        
    elif use_case == "V1_withClaim":
        original_models_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V1_withClaim_predictRenewal"
        model_path_pretrained = original_models_path + "\\grid"
        data_file = "V1_WithClaim_RenewalPredict_Data.csv"
        no_of_models = 6
        
    elif use_case == "V2_noClaim":
        original_models_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V2_noClaim_predictRenewal"
        model_path_pretrained = original_models_path + "\\grid"
        data_file = "V2_NoClaim_RenewalPredict_Data.csv"
        no_of_models = 10
        
    elif use_case == "V2_withClaim":
        original_models_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V2_withClaim_predictRenewal_63"
        model_path_pretrained = original_models_path + "\\grid"
        data_file = "V2_WithClaim_RenewalPredict_Data_63.csv"
        no_of_models = 10
    
    else:
        print("Incorrect Use case!")
        
    return original_models_path, model_path_pretrained, data_file, no_of_models

############################################################################
#read an already trained model
def read_model(model_arch_num,model_num,model_path):
    os.chdir(model_path)
    file_name = "Model_" + str(model_arch_num) + "_" + str(model_num)
    
    # load json and create model
    json_file = open(file_name + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights(file_name + ".h5")
#    print("Loaded model from disk")
    print(str(model_arch_num) + " - " +str(model_num))
    
    return loaded_model 

def getMaxAccuracy(file_name):
    plot_info_file = pd.read_csv(file_name)
    max_accuracy_value = max(plot_info_file['accuracy'])
    
    return max_accuracy_value

########################################################################
#getting initial learning rate
def storeParameters(params):
    with open('last_grid_params.pickle', 'wb') as handle:
        pickle.dump(params, handle)

def getInitialLearningRate():
    with open('last_grid_params.pickle', 'rb') as handle:
        initial_params = pickle.load(handle)
    return initial_params['learning_rate']

# define step decay function
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print('lr:', step_decay(len(self.losses)))

def step_decay(epoch):
    initial_lrate = getInitialLearningRate()
#    keras.backend.eval(optimizer.lr)
#    print("Initial learning rate = " + str(initial_lrate))
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate

# define exponential decay function
class LossHistory_(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(exp_decay(len(self.losses)))
        print('lr:', exp_decay(len(self.losses)))

def exp_decay(epoch):
    initial_lrate = getInitialLearningRate()
    k = 0.1
    lrate = initial_lrate * np.exp(-k*epoch)
    return lrate

############################################################################
#different model architecture based on model numbers
def create_model(use_case,model_num,optimizer,dropout=False,metric=['accuracy']):
    if use_case == "V2_noClaim" or use_case == "V2_withClaim":
        if model_num == 1:
            model_1 = Sequential()
        
            #Convolutional Layers
            model_1.add(Reshape((62, 1), input_shape=(62,)))
            model_1.add(Conv1D(31, kernel_size=2, strides=2, padding="same", activation = 'relu'))
            
            #Dense Layers
            model_1.add(Flatten())
            model_1.add(Dense(units=15, activation='relu'))
            if dropout is not False:
                model_1.add(Dropout(dropout))
            model_1.add(Dense(units=1, activation='sigmoid'))
            
            model_1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metric)
            
            return model_1
        
        elif model_num == 2:
            model_2 = Sequential()
    
            #Convolutional Layers
            model_2.add(Reshape((62, 1), input_shape=(62,)))
            model_2.add(Conv1D(60, kernel_size=3, strides=1, padding="same", activation = 'relu'))
            model_2.add(Conv1D(30, kernel_size=2, strides=2, padding="same", activation = 'relu'))
            
            #Dense Layers
            model_2.add(Flatten())
            model_2.add(Dense(units=15, activation='relu'))
            if dropout is not False:
                model_2.add(Dropout(dropout))
            model_2.add(Dense(units=1, activation='sigmoid'))
            
            model_2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
            return model_2
            
        elif model_num == 3:
            model_3 = Sequential()
            
            #Convolutional Layers
            model_3.add(Reshape((62, 1), input_shape=(62,)))
            model_3.add(Conv1D(61, kernel_size=2, strides=1, padding="same", activation = 'relu'))
            model_3.add(Conv1D(60, kernel_size=2, strides=1, padding="same", activation = 'relu'))
            model_3.add(Conv1D(30, kernel_size=2, strides=2, padding="same", activation = 'relu'))
            
            #Dense Layers
            model_3.add(Flatten())
            model_3.add(Dense(units=15, activation='relu'))
            if dropout is not False:
                model_3.add(Dropout(dropout))
            model_3.add(Dense(units=1, activation='sigmoid'))
            
            model_3.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
            return model_3
            
        elif model_num == 4:
            model_4 = Sequential()
            
            #Convolutional Layers
            model_4.add(Reshape((62, 1), input_shape=(62,)))
            model_4.add(Conv1D(60, kernel_size=3, strides=1, padding="same", activation = 'relu'))
            model_4.add(Conv1D(30, kernel_size=2, strides=2, padding="same", activation = 'relu'))
            
            #Dense Layers
            model_4.add(Flatten())
            model_4.add(Dense(units=15, activation='relu'))
            if dropout is not False:
                model_4.add(Dropout(dropout))
            model_4.add(Dense(units=10, activation='relu'))
            if dropout is not False:
                model_4.add(Dropout(dropout))
            model_4.add(Dense(units=5, activation='relu'))
            if dropout is not False:
                model_4.add(Dropout(dropout))
            model_4.add(Dense(units=1, activation='sigmoid'))
            
            model_4.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
          
            return model_4
            
        elif model_num == 5:
            model_5 = Sequential()
    
            #Convolutional Layers
            model_5.add(Reshape((62, 1), input_shape=(62,)))
            model_5.add(Conv1D(31, kernel_size=2, strides=2, padding="same", activation = 'relu'))
            model_5.add(Conv1D(30, kernel_size=2, strides=1, padding="same", activation = 'relu'))
            
            #Dense Layers
            model_5.add(Flatten())
            model_5.add(Dense(units=15, activation='relu'))
            if dropout is not False:
                model_5.add(Dropout(dropout))
            model_5.add(Dense(units=1, activation='sigmoid'))
            
            model_5.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
            return model_5
            
        elif model_num == 6:
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
            if dropout is not False:
                model_6.add(Dropout(dropout))
            model_6.add(Dense(units=1, activation='sigmoid'))
            
            model_6.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
            return model_6
            
        elif model_num == 7:
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
            if dropout is not False:
                model_7.add(Dropout(dropout))
            model_7.add(Dense(units=15, activation='relu'))
            if dropout is not False:
                model_7.add(Dropout(dropout))
            model_7.add(Dense(units=1, activation='sigmoid'))
            
            model_7.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
            return model_7
            
        elif model_num == 8:
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
            if dropout is not False:
                model_8.add(Dropout(dropout))
            model_8.add(Dense(units=1, activation='sigmoid'))
            
            model_8.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
           
            return model_8
            
        elif model_num == 9:
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
            if dropout is not False:
                model_9.add(Dropout(dropout))
            model_9.add(Dense(units=1, activation='sigmoid'))
            
            model_9.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            
            return model_9
            
        elif model_num == 10:
            model_10 = Sequential()
    
            #Convolutional Layers
            model_10.add(Reshape((62, 1), input_shape=(62,)))
            model_10.add(Conv1D(31, kernel_size=2, strides=2, padding="same", activation = 'relu'))
            model_10.add(Conv1D(20, kernel_size=11, strides=1, padding="same", activation = 'relu'))
            
            #Dense Layers
            model_10.add(Flatten())
            model_10.add(Dense(units=30, activation='relu'))
            if dropout is not False:
                model_10.add(Dropout(dropout))
            model_10.add(Dense(units=1, activation='sigmoid'))
            
            model_10.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])      
            
            return model_10
            
        else:
            print("Wrong model num : architecture not defined")
            
    elif use_case == "V1_withClaim" or use_case == "V1_noClaim":
        if model_num == 1:
            model_1 = Sequential()

            #Convolutional Layers
            model_1.add(Reshape((97, 1), input_shape=(97,)))
            model_1.add(Conv1D(20, kernel_size=5, strides=5, padding="same", activation = 'relu'))
            
            #Dense Layers
            model_1.add(Flatten())
            model_1.add(Dense(units=30, activation='relu'))
            if dropout is not False:
                model_1.add(Dropout(dropout))
            model_1.add(Dense(units=15, activation='relu'))
            if dropout is not False:
                model_1.add(Dropout(dropout))
            model_1.add(Dense(units=1, activation='sigmoid'))
            
            model_1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            
            return model_1

        elif model_num == 2:
            model_2 = Sequential()

            #Convolutional Layers
            model_2.add(Reshape((97, 1), input_shape=(97,)))
            model_2.add(Conv1D(48, kernel_size=3, strides=2, padding="same", activation = 'relu'))
            model_2.add(Conv1D(20, kernel_size=2, strides=2, padding="same", activation = 'relu'))
            
            #Dense Layers
            model_2.add(Flatten())
            model_2.add(Dense(units=30, activation='relu'))
            if dropout is not False:
                model_2.add(Dropout(dropout))
            model_2.add(Dense(units=15, activation='relu'))
            if dropout is not False:
                model_2.add(Dropout(dropout))
            model_2.add(Dense(units=1, activation='sigmoid'))
            
            model_2.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            return model_2
        
        elif model_num == 3:
            model_3 = Sequential()

            #Convolutional Layers
            model_3.add(Reshape((97, 1), input_shape=(97,)))
            model_3.add(Conv1D(48, kernel_size=2, strides=2, padding="same", activation = 'relu'))
            model_3.add(Conv1D(40, kernel_size=8, strides=1, padding="same", activation = 'relu'))
            model_3.add(Conv1D(20, kernel_size=2, strides=2, padding="same", activation = 'relu'))
            
            #Dense Layers
            model_3.add(Flatten())
            model_3.add(Dense(units=30, activation='relu'))
            if dropout is not False:
                model_3.add(Dropout(dropout))
            model_3.add(Dense(units=15, activation='relu'))
            if dropout is not False:
                model_3.add(Dropout(dropout))
            model_3.add(Dense(units=1, activation='sigmoid'))
            
            model_3.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            
            return model_3
        
        elif model_num == 4:
            model_4 = Sequential()

            #Convolutional Layers
            model_4.add(Reshape((97, 1), input_shape=(97,)))
            model_4.add(Conv1D(25, kernel_size=4, strides=4, padding="same", activation = 'relu'))
            
            #Dense Layers
            model_4.add(Flatten())
            model_4.add(Dense(units=30, activation='relu'))
            if dropout is not False:
                model_4.add(Dropout(dropout))
            model_4.add(Dense(units=15, activation='relu'))
            if dropout is not False:
                model_4.add(Dropout(dropout))
            model_4.add(Dense(units=1, activation='sigmoid'))
            
            model_4.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            
            return model_4
        
        elif model_num == 5:
            model_5 = Sequential()

            #Convolutional Layers
            model_5.add(Reshape((97, 1), input_shape=(97,)))
            model_5.add(Conv1D(33, kernel_size=3, strides=3, padding="same", activation = 'relu'))
            model_5.add(Conv1D(20, kernel_size=14, strides=1, padding="same", activation = 'relu'))
            
            #Dense Layers
            model_5.add(Flatten())
            model_5.add(Dense(units=30, activation='relu'))
            if dropout is not False:
                model_5.add(Dropout(dropout))
            model_5.add(Dense(units=15, activation='relu'))
            if dropout is not False:
                model_5.add(Dropout(dropout))
            model_5.add(Dense(units=1, activation='sigmoid'))
            
            model_5.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            
            return model_5

        elif model_num == 6:
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
            if dropout is not False:
                model_6.add(Dropout(dropout))
            model_6.add(Dense(units=15, activation='relu'))
            if dropout is not False:
                model_6.add(Dropout(dropout))
            model_6.add(Dense(units=1, activation='sigmoid'))
            
            model_6.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            
            return model_6
        
        else:
            print("Wrong model num : architecture not defined")

    else:
        print("Wrong Use case!")
    
############################################################################
#Modelling with grid parameters
os.getcwd()

grid = {'nesterov': [True],
        'batch_size': [20,30],
        'epochs': [20,50,100,150],
        'learning_rate': [0.1, 0.2, 0.3],
        'momentum': [0.5,0.65,0.8],
        'decay': ['decay','step','exponential'],
        'dropout' : [False, 0.25]
        }

#grid = {'nesterov': [True],
#        'batch_size': [20,30],
#        'epochs': [100,200,500],
#        'learning_rate': [0.1, 0.2, 0.3, 0.4],
#        'momentum': [0.5,0.6,0.7,0.8],
#        'decay': ['decay','step','exponential'],
#        'dropout' : [False, 0.2, 0.3]
##        'optimizers': ['adam','adagrad','adadelta','RMSprop']
#        }

param_grid = list(ParameterGrid(grid))

#Modelling
data_file_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Data"
Insurance_use_cases = ["V1_noClaim","V1_withClaim","V2_noClaim","V2_withClaim"]

use_case = "V1_noClaim"
for use_case in Insurance_use_cases:
    original_models_path, model_path_grid, data_file, no_of_models = getUseCaseDataInfo(use_case)
    
    #Read Data
    os.chdir(data_file_path)

    full_data = pd.read_csv(data_file)
    
    data_0 = full_data.copy(deep=True)
    data_0_columns = data_0.columns.tolist()
    
    scaler = MinMaxScaler(feature_range=(0,1))
    data_0_scaled = scaler.fit_transform(data_0)
    
    data_0 = pd.DataFrame(data_0_scaled, columns=data_0_columns)
    
    #train test split
    y = data_0["Renewed"]
    X = data_0.drop(["Renewed"],axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)
    
    X_train = X_train.values
    X_test = X_test.values
    
    y_train = y_train.values
    y_test = y_test.values
    
    model_num = 6
#    for model_num in range(1,no_of_models+1,1):
    for model_num in range(5,no_of_models+1,1):
        model_path_grid_all = model_path_grid + "\\model_" + str(model_num) + "\\grid_all"
        model_path_grid_selected = model_path_grid + "\\model_" + str(model_num) + "\\grid_selected"
        
        os.chdir(model_path_grid_all)
        
        models_grid = []
        max_accuracy_all_grid = []
        
        num_models_already_created = 87
        for read_models_index in range(0,num_models_already_created,1):
            model = read_model(model_num, read_models_index, model_path_grid_all)
            plot_info_file_name = "Model_" + str(model_num) + "_" + str(read_models_index) + "_all_plot_info.csv"
            model_max_accuracy = getMaxAccuracy(plot_info_file_name)
            
            models_grid.append(model)
            max_accuracy_all_grid.append(model_max_accuracy)

#        num_models_already_created = 0
        
        for index in range(num_models_already_created,len(param_grid)):
            params = param_grid[index]
            storeParameters(params)
        
            if params['decay'] == 'decay':
                #Assign parameters
                batch_size = params['batch_size']
                epochs = params['epochs']
                learning_rate = params['learning_rate']
                decay_rate = learning_rate / epochs
                momentum = params['momentum']
                nesterov = params['nesterov']
                dropout = params['dropout']
                
                #Create the optimizer function
                optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=nesterov)
                
                #create, compile and append model to list
                model = create_model(use_case, model_num,optimizer,dropout)
                models_grid.append(model)
                
                #fit the model
                model = model.fit(X_train, y_train, 
                                  epochs=epochs, 
                                  batch_size=batch_size)
            
            elif params['decay'] == 'step':
                #Assign parameters
                batch_size = params['batch_size']
                epochs = params['epochs']
                learning_rate = params['learning_rate']
                momentum = params['momentum']
                nesterov = params['nesterov']
                dropout = params['dropout']
        
                #Create the optimizer function
                optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=0.0, nesterov=nesterov)
                
                #create, compile and append model to list
                model = create_model(use_case, model_num,optimizer,dropout)
                models_grid.append(model)
                
                # learning schedule callback
                loss_history = LossHistory()
                lrate = LearningRateScheduler(step_decay)
                callbacks_list = [loss_history, lrate]
                
                #fit the model
                model.fit(X_train, y_train,
                          epochs=epochs, 
                          batch_size=batch_size,
                          callbacks=callbacks_list)
                
            elif params['decay'] == 'exponential':
                #Assign parameters
                batch_size = params['batch_size']
                epochs = params['epochs']
                learning_rate = params['learning_rate']
                momentum = params['momentum']
                nesterov = params['nesterov']
                dropout = params['dropout']
        
                #Create the optimizer function
                optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum, decay=0.0, nesterov=nesterov)
                
                #create, compile and append model to list
                model = create_model(use_case, model_num,optimizer,dropout)
                models_grid.append(model)
                
                # learning schedule callback
                loss_history_ = LossHistory_()
                lrate_ = LearningRateScheduler(exp_decay)
                callbacks_list_ = [loss_history_, lrate_]
                
                #fit the model
                model = model.fit(X_train, y_train, 
                                  epochs=epochs, 
                                  batch_size=batch_size, 
                                  callbacks=callbacks_list_)
            else:
                print("Incorrect decay function type")
            
            #for all models    
            #saving model info with respect to threshold
            file_name = "Model_" + str(model_num) + "_" + str(index)
            file_path = model_path_grid_all
            
            max_accuracy = plotMetricsWithThreshold(models_grid[index], X_test, y_test, fig_name = file_name, fig_path = file_path,
                                                    display_info=False,
                                                    return_max_metric='accuracy')
            max_accuracy_all_grid.append(max_accuracy)
            
            #saving model and model info
            saveModelandModelInfo(models_grid[index], file_name, file_path)
        
        
        #Get top 5 models by max accuracy
        max_accuracy_array = np.array(max_accuracy_all_grid)
        top_n_by_accuracy = max_accuracy_array.argsort()[-5:][::-1]
        
        #Copy those files in folder "grid_selected" with their rank
        for rank_index in range(len(top_n_by_accuracy)):
            model_index = top_n_by_accuracy[rank_index]
            
            #Read model info
            #Model is not being read again as it is already stored in list "models_grid"
            os.chdir(model_path_grid_all)
            original_file_name = "Model_" + str(model_num) + "_" + str(model_index) + "_all_plot_info"
            threshold_metrics_info = pd.read_csv(original_file_name + ".csv")
            
            #Save model and model info in folder "grid_selected" with nomenclature as 
            #<old_file_name>_accuracy_<rank>.csv
            os.chdir(model_path_grid_selected)
            new_file_name = "Model_" + str(model_num) + "_" + str(model_index) + "_accuracy_" + str(rank_index)
            threshold_metrics_info.to_csv(new_file_name + ".csv", index=False)
            saveModelandModelInfo(models_grid[model_index], new_file_name, model_path_grid_selected)
            
#################################################################

# =============================================================================
# use_case = "V2_noClaim"
# original_models_path, model_path_grid, data_file, no_of_models = getUseCaseDataInfo(use_case)
# 
# for arch_num in range(1,no_of_models+1):
#     models_path = model_path_grid + "\\model_" + str(arch_num) + "\\grid_selected"
#     threshold_files_path = models_path + "\\Threshold"
#     
#     for root,dirs,files in os.walk(models_path):
#         for file_name in files:
#            if file_name.endswith(".csv"):
#                report_file = pd.read_csv(root + "\\" + file_name)
#                
#                file_name = file_name[0:-4]
#                threshold_file_name = file_name + "_threshold" 
#                 
#                filterThresholds(file_name, models_path, 
#                threshold_file_name, threshold_files_path,
#                save_separate_files = True,
#                accuracy_filter = 0.6, precision_filter = 0.75, recall_filter = 0.75, 
#                TPR_filter = 0.6, FPR_filter = None, TNR_filter = 0.6,
#                fpoint5score_filter = None, f1score_filter = [0.5,0.5], f2score_filter = None,
#                precision_class = "majority", recall_class = "minority",
#                fpoint5score_class = None, f1score_class = "both", f2score_class = None
#                )
#     print(arch_num)
# 
# =============================================================================

# =============================================================================
# combineModelReports(original_models_path, "grid",
#                         "V2_noClaim_CNN_Report_1", original_models_path,
#                         no_of_models)
# 
# =============================================================================






# =============================================================================
# filter_val
# filterThresholds(file_name, models_path, 
#                  threshold_file_name, threshold_files_path,
#                  save_separate_files = True,
#                  accuracy_filter = 0.7, precision_filter = 0.75, recall_filter = 0.75, 
#                  TPR_filter = 0.6, FPR_filter = None, TNR_filter = 0.6,
#                  fpoint5score_filter = None, f1score_filter = 0.5, f2score_filter = None,
#                  precision_class = "majority", recall_class = "minority",
#                  fpoint5score_class = None, f1score_class = "both", f2score_class = None
#                  )
# 
# =============================================================================
