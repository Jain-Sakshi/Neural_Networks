# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 18:44:12 2019

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
os.getcwd()

use_case = "V2_withClaim"
original_models_path, model_path_grid, data_file, no_of_models = getUseCaseDataInfo(use_case)

for arch_num in range(1,no_of_models+1):
    models_path_pretrained = original_models_path + "\\pretrained"
    models_path = models_path_pretrained + "\\model_" + str(arch_num) + "\\grid_selected"
    threshold_files_path = models_path
    
    for root,dirs,files in os.walk(models_path):
        for file_name in files:
#           if file_name.endswith(".csv"):
            if file_name.startswith("Model_") and file_name.endswith(".csv") and "threshold" not in file_name:
               report_file = pd.read_csv(root + "\\" + file_name)
               
               model_file_name = file_name[0:-4]
               threshold_file_name = model_file_name + "_threshold" 
                
               filterThresholds(model_file_name, models_path, 
               threshold_file_name, threshold_files_path,
               save_separate_files = False,
               accuracy_filter = 0.75, precision_filter = 0.85, recall_filter = 0.75, 
               TPR_filter = 0.75, FPR_filter = None, TNR_filter = 0.76,
               fpoint5score_filter = None, f1score_filter = [0.7, 0.8], f2score_filter = None,
               precision_class = "majority", recall_class = "minority",
               fpoint5score_class = None, f1score_class = "both", f2score_class = None
               )
    print(arch_num)

############################################################################
def appendResults(threshold_result_file, use_case_summary_df, model_num, arch_num):
    temp_report_file = threshold_result_file
    temp_report_file["Model_Num"] = model_num
    temp_report_file["Arch_Num"] = arch_num
    
    use_case_summary_df = use_case_summary_df.append(temp_report_file)
    
    return use_case_summary_df

def getModelNum(file_name):
    file_name_split = file_name.split("_")
    model_num = file_name_split[2]
    
    return model_num

use_case_summary_df = pd.DataFrame()

for arch_num in range(1,no_of_models+1):
    models_path = models_path_pretrained + "\\model_" + str(arch_num) + "\\grid_selected"
    threshold_files_path = models_path
    
    for root,dirs,files in os.walk(models_path):
        for file_name in files:
            if file_name.endswith("_threshold.csv"):
                model_num = getModelNum(file_name)
                threshold_result_file = pd.read_csv(root + "\\" + file_name)
                use_case_summary_df = appendResults(threshold_result_file, use_case_summary_df, model_num, arch_num)
                
os.getcwd()
os.chdir(original_models_path)
use_case_summary_df.to_csv(use_case + "_CNN_Results_Pretrained_Criteria_1.csv", index=False)
               
    
# =============================================================================
# for use case V2_noclaim (grid)

# Criteria 1
# TPR = 0.65
# TNR = 0.8
# fscore - 0.7
# accuracy = 0.7
# recall and precision (imbalanced) = 0.65 and 0.75


# Criteria 0
# TPR = 0.65
# TNR = 0.8
# fscore - 0.7
# accuracy = 0.75
# recall and precision (imbalanced) = 0.7 and 0.75
# 
# from initial values
# TPR = 0.6
# TNR = 0.6
# fscore - 0.5, 0.5
# accuracy = 0.7
# recall and precision (imbalanced) = 0.75 (how?)
# 
# =============================================================================
