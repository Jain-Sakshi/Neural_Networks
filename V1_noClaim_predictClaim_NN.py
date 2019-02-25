# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 10:18:04 2018

@author: sakshij
"""

import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

############################################################################
#def standardize(data, coln_name):
#    mean = np.mean(data[coln_name])
#    std_dev = np.std(data[coln_name])
#    
#    new_data_coln = (data[coln_name] - mean)/std_dev
#    
#    return new_data_coln
############################################################################
#Paths
model_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V1_predictClaim"
data_file_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Data"
data_file = "V1_ClaimPredict_Data_latest"

#os.getcwd()
#os.chdir('D:\\Data_Nik\\Vehicle1_Encoder_H')

############################################################################
#Report
os.chdir("D:\\Projects\\Tableau_Train_Combine\\NN\\Codes")
from Report_NN import report

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
data_0 = scaler.fit_transform(data_0)

#Data Prep
#for column_name in data_0.columns:
#    if max(data_0[column_name]) > 1:
#        scaler = MinMaxScaler(feature_range=(0,1))
#        data_0[column_name] = scaler.fit_transform(data_0[column_name])

############################################################################
#Train Test Split
h2o.init(nthreads = -1, max_mem_size = 6)

data_0_h2o = h2o.H2OFrame(data_0)
data_0_h2o.columns = data_0_columns
data_0_h2o.shape
data_0_h2o_shape = data_0_h2o.shape

#data = data.drop(["Unnamed: 0"],axis=1)

data_0_h2o['ClaimStatus'] = data_0_h2o['ClaimStatus'].asfactor()  #encode the binary repsonse as a factor
data_0_h2o['ClaimStatus'].levels()

splits = data_0_h2o.split_frame(ratios=[0.65, 0.00], seed=1)

train = splits[0]
valid = splits[1]
test = splits[2]

train.nrow
valid.nrow
test.nrow

y = 'ClaimStatus'
x = list(data_0_h2o.columns)
x.remove('ClaimStatus')
    
############################################################################
#Modelling

# =============================================================================
# These arguments remain the same for every model
# export_weights_and_biases=True,
# shuffle_training_data=True,
# reproducible=True
# 
# =============================================================================
os.chdir(model_path)
os.getcwd()

#NN1
dl_fit1 = H2ODeepLearningEstimator(model_id='dl_fit1', 
                                   epochs=20, 
                                   hidden=[20,10,5],
                                   seed=1,
                                   max_categorical_features=10,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
                                   )
dl_fit1.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward"
defined_params = "(model_id='dl_fit1', epochs=20, hidden=[20,10,5], seed=1, max_categorical_features=10)"
file_name = "NN_1"
training_ratio = 0.65

report_df = report(dl_fit1, test, y , model_path, 4, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=None, sampling=None, model_type='basic')

#NN2
dl_fit2 = H2ODeepLearningEstimator(model_id='dl_fit2', 
                                   epochs=20, 
                                   hidden=[20,10,5],
                                   seed=1,
                                   max_categorical_features=10,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
                                   adaptive_rate=False,
                                   rate_annealing=0.2
                                   )
dl_fit2.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward"
defined_params = "(model_id='dl_fit2', epochs=20, hidden=[20,10,5], seed=1, max_categorical_features=10, adaptive_rate=False, rate_annealing=0.2)"
file_name = "NN_2"
training_ratio = 0.65
num_layers = 4

report_df = report(dl_fit2, test, y , model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=None, sampling=None, model_type='basic')

#NN3
dl_fit3 = H2ODeepLearningEstimator(model_id='dl_fit3', 
                                   epochs=20, 
                                   hidden=[20,10,5],
                                   seed=1,
                                   max_categorical_features=10,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
                                   adaptive_rate=False,
                                   rate_annealing=0.5
                                   )
dl_fit3.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward"
defined_params = "(model_id='dl_fit3', epochs=20, hidden=[20,10,5], seed=1, max_categorical_features=10, adaptive_rate=False, rate_annealing=0.5)"
file_name = "NN_3"
training_ratio = 0.65
num_layers = 4

report_df = report(dl_fit3, test, y , model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=None, sampling=None, model_type='basic')

#NN4
dl_fit4 = H2ODeepLearningEstimator(model_id='dl_fit4', 
                                   epochs=20, 
                                   hidden=[20,10,5],
                                   seed=1,
                                   max_categorical_features=10,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
                                   adaptive_rate=False,
                                   rate_annealing=0.8
                                   )
dl_fit4.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward"
defined_params = "(model_id='dl_fit4', epochs=20, hidden=[20,10,5], seed=1, max_categorical_features=10, adaptive_rate=False, rate_annealing=0.8)"
file_name = "NN_4"
training_ratio = 0.65
num_layers = 4

report_df = report(dl_fit4, test, y , model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=None, sampling=None, model_type='basic')

#NN5
dl_fit5 = H2ODeepLearningEstimator(model_id='dl_fit6', 
                                   epochs=20, 
                                   hidden=[30,20,10,5],
                                   seed=1,
                                   max_categorical_features=10,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
#                                   adaptive_rate=False,
#                                   rate_annealing=0.7
                                   )
dl_fit5.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward"
defined_params = "(model_id='dl_fit5', epochs=20, hidden=[30,20,10,5], seed=1, max_categorical_features=10)"
file_name = "NN_5"
training_ratio = 0.65
num_layers = 4

report_df = report(dl_fit5, test, y , model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=None, sampling=None, model_type='basic')

#NN6
dl_fit6 = H2ODeepLearningEstimator(model_id='dl_fit6', 
                                   epochs=20, 
                                   hidden=[30,15,5],
                                   seed=1,
                                   max_categorical_features=10,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
#                                   adaptive_rate=False,
#                                   rate_annealing=0.7
                                   )
dl_fit6.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward"
defined_params = "(model_id='dl_fit6', epochs=20, hidden=[30,15,5], seed=1, max_categorical_features=10)"
file_name = "NN_6"
training_ratio = 0.65
num_layers = 4

report_df = report(dl_fit6, test, y , model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=None, sampling=None, model_type='basic')

#NN7
dl_fit7 = H2ODeepLearningEstimator(model_id='dl_fit7', 
                                   epochs=20, 
                                   hidden=[10,10,5],
                                   seed=1,
                                   max_categorical_features=10,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
#                                   adaptive_rate=False,
#                                   rate_annealing=0.7
                                   )
dl_fit7.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward"
defined_params = "(model_id='dl_fit7', epochs=20, hidden=[10,10,5], seed=1, max_categorical_features=10, adaptive_rate=False, rate_annealing=0.7)"
file_name = "NN_5"
training_ratio = 0.65
num_layers = 4

report_df = report(dl_fit7, test, y , model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=None, sampling=None, model_type='basic')

report_df.to_csv("Report_NN_v1_predictClaim.csv",index=False)