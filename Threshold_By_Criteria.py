# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 11:05:40 2019

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
from Filter_Thresholds import combineModelReports, filterThresholds, appendAllUseCaseFiles

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
#Creating a single file with all plot data of shortlisted models
Insurance_use_cases = ["V1_noClaim","V1_withClaim","V2_noClaim","V2_withClaim"]

for use_case in Insurance_use_cases:
    original_models_path, model_path_grid, data_file, no_of_models = getUseCaseDataInfo(use_case)
    all_data_file_name = use_case + "_all_shortlisted_models_data"
    appendAllUseCaseFiles(model_path_grid, no_of_models, all_data_file_name)

############################################################################
use_case = "V2_withClaim"
original_models_path, model_path_grid, data_file, no_of_models = getUseCaseDataInfo(use_case)

#Creating Threshold files for every shortlisted model for use_case
criteria_num = 0
for arch_num in range(1,no_of_models+1):
    models_path_grid = original_models_path + "\grid"
    models_path = models_path_grid + "\model_" + str(arch_num) + "\grid_selected\Threshold"
    
    for root,dirs,files in os.walk(models_path):
        for file_name in files:
#           if file_name.endswith(".csv"):
            if file_name.startswith("Model_") and file_name.endswith(".csv") and "threshold" not in file_name:
               report_file = pd.read_csv(root + "\" + file_name)
               
               model_file_name = file_name[0:-4]
               threshold_file_name = model_file_name + "_threshold" 
               
               filterThresholds(model_file_name, models_path, 
                                criteria_num, filtered_file_name = None, 
                                filtered_file_location = None,
                                save_separate_files = True,
                                accuracy_filter = None, precision_filter = None, recall_filter = None, 
                                TPR_filter = None, FPR_filter = None, TNR_filter = None,
                                fpoint5score_filter = None, f1score_filter = None, f2score_filter = None,
                                precision_class = None, recall_class = None,
                                fpoint5score_class = None, f1score_class = None, f2score_class = None,
                                fp_less_than_fn = None, append_fp_less_than_fn = False
                                )
               
###################################################################
#Creating Threshold files for every shortlisted model for use_case
criteria_num = 0
Insurance_use_cases = ["V1_noClaim","V1_withClaim","V2_noClaim","V2_withClaim"]

use_case = "V1_noClaim"
original_models_path, model_path_grid, data_file, no_of_models = getUseCaseDataInfo(use_case)

model_report_file_name = use_case + "_all_shortlisted_models_data"
model_report_file_location = model_path_grid
filtered_file_name = use_case + "_threshold_criteria_" + str(criteria_num)

accuracy = 0.75
precision = 0.8
recall = 0.55
tpr = 0.65
tnr = 0.8
f1score_filter = [0.8, 0.6]
precision_class = "majority"
recall_class = "1"
f1score_class = "both"
#f1score_filter = f1score_class = None

filterThresholds(model_report_file_name, model_report_file_location, 
                 criteria_num, filtered_file_name = filtered_file_name, 
                 filtered_file_location = model_report_file_location,
                 save_separate_files = False,
                 accuracy_filter = accuracy, precision_filter = precision, recall_filter = recall, 
                 TPR_filter = tpr, FPR_filter = None, TNR_filter = tnr,
                 fpoint5score_filter = None, f1score_filter = f1score_filter, f2score_filter = None,
                 precision_class = precision_class, recall_class = recall_class,
                 fpoint5score_class = None, f1score_class = f1score_class, f2score_class = None,
                 fp_less_than_fn = None, append_fp_less_than_fn = False
                 )
           