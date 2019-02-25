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

############################################################################
#Report
os.chdir("D:\\Projects\\Tableau_Train_Combine\\NN\\Codes")
from Report import report

cols = ['Training Sample','Features','Sampling','Classifier','Confusion Matrix','_','Threshold','Classification Report','Precision','Recall','F1-score','Support','__','Accuracy','TPR','FPR','Misclassification_Rate','Specificity','Total_Precision','Model_Filename','Grid_Params']
report_df = pd.DataFrame(columns=cols)

############################################################################
#defining paths and use case based variables
original_models_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V1_noClaim_predictRenewal"
model_path_tpot = original_models_path + "\\tpot"
data_file = "V1_NoClaim_RenewalPredict_Data"
no_of_models = 6
data_file_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Data"

#Reading the data
os.chdir(data_file_path)
data_df = pd.read_csv(data_file + ".csv")

data_df.dtypes

#encoder = LabelEncoder()
#target = encoder.fit_transform(data_df["Species"])
#data_df["Species"] = target
#
#y = data_df["Species"]
#X = data_df.drop(["Species"], axis=1)

y = data_df["Renewed"]
X = data_df.drop(["Renewed"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state = 0)

tpot = TPOTClassifier(generations=5, population_size=20, random_state = 0, verbosity = 3)
tpot.fit(X_train, y_train)

os.chdir(model_path_tpot)
tpot.export("tpot_pipeline_1.py")

tpot.score(X_test, y_test)

tpot.fitted_pipeline_
tpot.evaluated_individuals_
pareto_front_fitted_pipelines = tpot.pareto_front_fitted_pipelines_

grid_or_basic = 'basic'
file_name = 'tpot_1'
report_df = report(tpot,'tpot 1',X_test,y_test,0.35,grid_or_basic,file_name,report_df,parameters='generations=5, population_size=20, random_state = 0, verbosity = 3', save_file = False)

#TPOT 2

tpot_2 = TPOTClassifier(generations=10, population_size=20, random_state = 0, verbosity = 3)
tpot_2.fit(X_train, y_train)

os.chdir(model_path_tpot)
tpot_2.export("tpot_pipeline_2.py")

score_2 = tpot_2.score(X_test, y_test)

#tpot.fitted_pipeline_
#tpot.evaluated_individuals_
#pareto_front_fitted_pipelines = tpot.pareto_front_fitted_pipelines_

grid_or_basic = 'basic'
file_name = 'tpot_2'
report_df = report(tpot_2,'tpot 2',X_test,y_test,0.35,grid_or_basic,file_name,report_df,parameters='generations=10, population_size=20, random_state = 0, verbosity = 3', save_file = False)


#TPOT 3

tpot_3 = TPOTClassifier(generations=10, population_size=100, random_state = 0, verbosity = 3, max_eval_time_mins = 60)
tpot_3.fit(X_train, y_train)

os.chdir(model_path_tpot)
tpot_3.export("tpot_pipeline_3.py")

score_3 = tpot_3.score(X_test, y_test)

#tpot.fitted_pipeline_
#tpot.evaluated_individuals_
#pareto_front_fitted_pipelines = tpot.pareto_front_fitted_pipelines_

grid_or_basic = 'basic'
file_name = 'tpot_3'
report_df = report(tpot_3,'tpot 3',X_test,y_test,0.35,grid_or_basic,file_name,report_df,parameters='generations=10, population_size=20, random_state = 0, verbosity = 3, max_eval_time_mins = 60', save_file = False)

os.getcwd()
report_df.to_csv("Report_1.csv")

evaluated_pipelines_3 = tpot_3.evaluated_individuals_
evaluated_pipelines_3_list = list(evaluated_pipelines_3)

index = 0
for index_element in evaluated_pipelines_3.keys():
    if evaluated_pipelines_3[index_element]['generation'] != 0:
        print(evaluated_pipelines_3[index_element])
        print(index)
        break
    index+=1
    
    print(type(evaluated_pipelines_3[index_element]))
    
    

#TPOT 4

tpot_4 = TPOTClassifier(generations=10, population_size=100, random_state = 0, verbosity = 3, warm_start = True)
tpot_4.fit(X_train, y_train)

os.chdir(model_path_tpot)
tpot_4.export("tpot_pipeline_4.py")

score_4 = tpot_4.score(X_test, y_test)

#tpot.fitted_pipeline_
#tpot.evaluated_individuals_
#pareto_front_fitted_pipelines = tpot.pareto_front_fitted_pipelines_

grid_or_basic = 'basic'
file_name = 'tpot_4'
report_df = report(tpot_4,'tpot 4',X_test,y_test,0.35,grid_or_basic,file_name,report_df,parameters='generations=10, population_size=100, random_state = 0, verbosity = 3', save_file = False)



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