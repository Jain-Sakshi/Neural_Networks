# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 11:41:55 2018

@author: sakshij
"""

#==============================================================================
# Update the following packages:
#     keras - 2.2.4
#     tensorflow - 1.10+
# and then run
#==============================================================================

import numpy as np
from tensorflow.python.keras import backend as K
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

############################################################################
def getUseCaseDataInfo(use_case):
    if use_case == "V1_noClaim":
        original_models_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V1_noClaim_predictRenewal"
        model_path_pretrained = original_models_path + "\\pretrained"
        data_file = "V1_NoClaim_RenewalPredict_Data"
        no_of_models = 6
        
    elif use_case == "V1_withClaim":
        original_models_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V1_withClaim_predictRenewal"
        model_path_pretrained = original_models_path + "\\pretrained"
        data_file = "V1_WithClaim_RenewalPredict_Data"
        no_of_models = 6
        
    elif use_case == "V2_noClaim":
        original_models_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V2_noClaim_predictRenewal"
        model_path_pretrained = original_models_path + "\\pretrained"
        data_file = "V2_NoClaim_RenewalPredict_Data"
        no_of_models = 10
        
    elif use_case == "V2_withClaim":
        original_models_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V2_withClaim_predictRenewal_63"
        model_path_pretrained = original_models_path + "\\pretrained"
        data_file = "V2_WithClaim_RenewalPredict_Data_63"
        no_of_models = 10
    
    else:
        print("Incorrect Use case!")
            
    return original_models_path, model_path_pretrained, data_file, no_of_models
    
############################################################################
#different model architecture based on model numbers
def read_model(model_num, optimizer,model_path):
    os.chdir(model_path)
    file_name = "CNN_" + str(model_num)
    
    # load json and create model
    json_file = open(file_name + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # load weights into new model
    loaded_model.load_weights(file_name + ".h5")
    print("Loaded model from disk")
    
    loaded_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return loaded_model
        
############################################################################
#Modelling with grid parameters
os.getcwd()

grid = {'nesterov': [True, False],
        'epochs': [100,200,500],
        'batch_size' : [10,20,30],
        'learning_rate': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2],
#        'optimizers': ['adam','adagrad','adadelta','RMSprop']
        }

#temp_grid = {'nesterov': [True],
#        'epochs': [20],
#        'batch_size' : [10],
#        'learning_rate': [0.1, 0.2],
##        'optimizers': ['adam','adagrad','adadelta','RMSprop']
#        }

param_grid = list(ParameterGrid(grid))

#Modelling
data_file_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Data"
Insurance_use_cases = ["V1_noClaim","V1_withClaim","V2_noClaim","V2_withClaim"]

for use_case in Insurance_use_cases:
    original_models_path, model_path_pretrained, data_file, no_of_models = getUseCaseDataInfo(use_case)
    
    #Read Data
    os.chdir(data_file_path)

    full_data = pd.read_csv(data_file + ".csv")
    
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
    
    for model_num in range(1,no_of_models+1,1):
        model_path_pretrained_all = model_path_pretrained + "\\model_" + str(model_num) + "\\grid_all"
        model_path_pretrained_selected = model_path_pretrained + "\\model_" + str(model_num) + "\\grid_selected"
        
        os.chdir(model_path_pretrained_all)
        
        models_grid = []
        max_accuracy_all_grid = []
        
        for index in range(len(param_grid)):
            params = param_grid[index]
            
            batch_size = params['batch_size']
            epochs = params['epochs']
            learning_rate = params['learning_rate']
            decay_rate = learning_rate / epochs
            nesterov = params['nesterov']
            
            #Defining optimizer
            optimizer = keras.optimizers.SGD(lr=learning_rate, nesterov=nesterov)
            
            #Reading pretrained model
            model = read_model(model_num, optimizer, original_models_path)
            models_grid.append(model)
            
            #Re-fit and create a new model initialized with pretrained model weights  
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        
            #saving model info with respect to threshold
            file_name = "Model_" + str(model_num) + "_" + str(index)
            file_path = model_path_pretrained_all
            
            #storng max accuracy possible with this model (for shortlisting)
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
            os.chdir(model_path_pretrained_all)
            original_file_name = "Model_" + str(model_num) + "_" + str(model_index) + "_all_plot_info"
            threshold_metrics_info = pd.read_csv(original_file_name + ".csv")
            
            #Save model and model info in folder "grid_selected" with nomenclature as 
            #<old_file_name>_accuracy_<rank>.csv
            os.chdir(model_path_pretrained_selected)
            new_file_name = "Model_" + str(model_num) + "_" + str(model_index) + "_accuracy_" + str(rank_index)
            threshold_metrics_info.to_csv(new_file_name + ".csv", index=False)
            saveModelandModelInfo(models_grid[model_index], new_file_name, model_path_pretrained_selected)