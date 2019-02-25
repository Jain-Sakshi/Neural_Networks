# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 15:00:34 2018

@author: sakshij
"""

import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from h2o.grid.grid_search import H2OGridSearch

os.chdir("D:\\Projects\\Tableau_Train_Combine\\NN\\Codes")
from Plot_Metrics import plotMetricsWithThreshold

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
model_path_63 = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V2_withClaim_predictRenewal_63"
data_file_path_63 = "D:\\Projects\\Tableau_Train_Combine\\NN\\Data"
data_file_63 = "V2_WithClaim_RenewalPredict_Data_63"

############################################################################
#Report
os.chdir("D:\\Projects\\Tableau_Train_Combine\\NN\\Codes")
from Report_NN import report

cols = ['Training Sample','Features','Sampling','Classifier','Confusion Matrix','_','Threshold','Classification Report','Precision','Recall','F1-score','Support','__','Accuracy','TPR','FPR','Misclassification_Rate','Specificity','Total_Precision','Model_Filename','Grid_Params']
report_df = pd.DataFrame(columns=cols)

############################################################################
#Data        
os.chdir(data_file_path_63)

full_data = pd.read_csv(data_file_63 + ".csv")

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

data_0_h2o['Renewed'] = data_0_h2o['Renewed'].asfactor()  #encode the binary repsonse as a factor
data_0_h2o['Renewed'].levels()

splits = data_0_h2o.split_frame(ratios=[0.65, 0.00], seed=1)

train = splits[0]
valid = splits[1]
test = splits[2]

train.nrow
valid.nrow
test.nrow

y = 'Renewed'
x = list(data_0_h2o.columns)
x.remove('Renewed')

y_test_all = test['Renewed'].as_data_frame()
y_test = np.array(y_test_all['Renewed'])
    
############################################################################
#Data Distribution
pd.value_counts(train['Renewed'].as_data_frame()['Renewed'])
#1    272
#0    178
pd.value_counts(test['Renewed'].as_data_frame()['Renewed'])
#1    154
#0     81
############################################################################
#Modelling

# =============================================================================
# These arguments remain the same for every model
# export_weights_and_biases=True,
# shuffle_training_data=True,
# reproducible=True
# 
# =============================================================================
os.chdir(model_path_63)
os.getcwd()

#First basic models would be run
#Following that, a few would be shortlisted (on the basis of
#network architecture) and would be run with a grid

#NN1
dl_fit1 = H2ODeepLearningEstimator(model_id='dl_fit1',
                                   hidden=[30,15],
                                   seed=1,
                                   max_categorical_features=4,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
                                   adaptive_rate=True,
                                   )

dl_fit1.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward"
defined_params = ""
file_name = "NN_1"
training_ratio = 0.65
num_layers = 2

os.getcwd()

fig_name = file_name + "_plot"
plot_info_1_base = plotMetricsWithThreshold(dl_fit1, test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True, model_type='h2o')

threshold = 0.64
report_df = report(dl_fit1, test, y , model_path_63, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=threshold, sampling=None, model_type='basic')

#NN2
dl_fit2 = H2ODeepLearningEstimator(model_id='dl_fit2',
                                   hidden=[20,30,15],
                                   seed=1,
                                   max_categorical_features=4,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
                                   adaptive_rate=True,
                                   )

dl_fit2.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward"
defined_params = ""
file_name = "NN_2"
training_ratio = 0.65
num_layers = 3

os.getcwd()

fig_name = file_name + "_plot"
plot_info_1_base = plotMetricsWithThreshold(dl_fit2, test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True, model_type='h2o')

threshold = 0.68
report_df = report(dl_fit2, test, y , model_path_63, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=threshold, sampling=None, model_type='basic')

#NN3
dl_fit3 = H2ODeepLearningEstimator(model_id='dl_fit3',
                                   hidden=[50,20,30,15],
                                   seed=1,
                                   max_categorical_features=4,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
                                   adaptive_rate=True,
                                   )

dl_fit3.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward"
defined_params = ""
file_name = "NN_3"
training_ratio = 0.65
num_layers = 4

os.getcwd()

fig_name = file_name + "_plot"
plot_info_1_base = plotMetricsWithThreshold(dl_fit3, test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True, model_type='h2o')

threshold = 0.71
report_df = report(dl_fit3, test, y , model_path_63, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=threshold, sampling=None, model_type='basic')

#report_df = report_df.drop(report_df.index[[6,7,8]])

#NN4
dl_fit4 = H2ODeepLearningEstimator(model_id='dl_fit4',
                                   hidden=[50,40,30,20],
                                   seed=1,
                                   max_categorical_features=4,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
                                   adaptive_rate=True,
                                   )

dl_fit4.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward"
defined_params = ""
file_name = "NN_4"
training_ratio = 0.65
num_layers = 4

os.getcwd()

fig_name = file_name + "_plot"
plot_info_1_base = plotMetricsWithThreshold(dl_fit4, test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True, model_type='h2o')

threshold = 0.7
report_df = report(dl_fit4, test, y , model_path_63, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=threshold, sampling=None, model_type='basic')

#NN5
dl_fit5 = H2ODeepLearningEstimator(model_id='dl_fit5',
                                   hidden=[50,40,20,30,15,10],
                                   seed=1,
                                   max_categorical_features=4,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
                                   adaptive_rate=True,
                                   )

dl_fit5.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward"
defined_params = ""
file_name = "NN_5"
training_ratio = 0.65
num_layers = 6

os.getcwd()

fig_name = file_name + "_plot"
plot_info_1_base = plotMetricsWithThreshold(dl_fit5, test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True, model_type='h2o')

threshold = 0.67
report_df = report(dl_fit5, test, y , model_path_63, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=threshold, sampling=None, model_type='basic')

#NN6
dl_fit6 = H2ODeepLearningEstimator(model_id='dl_fit6',
                                   hidden=[60,50,40,20,30,15,10,5],
                                   seed=1,
                                   max_categorical_features=4,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
                                   adaptive_rate=True,
                                   )

dl_fit6.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward"
defined_params = ""
file_name = "NN_6"
training_ratio = 0.65
num_layers = 8

os.getcwd()

fig_name = file_name + "_plot"
plot_info_1_base = plotMetricsWithThreshold(dl_fit6, test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True, model_type='h2o')

threshold = 0.79
report_df = report(dl_fit6, test, y , model_path_63, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=threshold, sampling=None, model_type='basic')

path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V2_withClaim_predictRenewal_63"
os.chdir(path)
os.getcwd()

names_list = ["dl_fit1","dl_fit2","dl_fit3","dl_fit4","dl_fit5","dl_fit6"]
num_layers = [2,3,4,4,6,8]
thresholds = [0.64 ,0.68 ,0.71 ,0.7 ,0.67 ,0.79]

for index in range(len(names_list)):
    model = h2o.load_model(path + "\\" + names_list[index])
    num_layer = num_layers[index]
    threshold = thresholds[index]
    
    model_name = "MLP - Feedforward"
    defined_params = ""
    file_name = "NN_" + str(index+1)
    training_ratio = 0.65
    
    report_df = report(model, test, y , model_path_63, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=threshold, sampling=None, model_type='basic',
              store_model_perf = False, store_model=False, store_model_weights = False)
    
report_df.to_csv("Report_NN_V2_withClaim_predictRenewal_63cols_1.csv",index=False)