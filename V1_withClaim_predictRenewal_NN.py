# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 13:16:16 2018

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
model_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V1_withClaim_predictRenewal"
data_file_path = "D:\\Projects\\Tableau_Train_Combine\\NN\\Data"
data_file = "V1_WithClaim_RenewalPredict_Data"

############################################################################
#Report
os.chdir("D:\\Projects\\Tableau_Train_Combine\\NN\\Codes")
from Report_NN import report

cols = ['Training Sample','Features','Sampling','Classifier','Confusion Matrix','_','Threshold','Classification Report','Precision','Recall','F1-score','Support','__','Accuracy','TPR','FPR','Misclassification_Rate','Specificity','Total_Precision','Model_Filename','Grid_Params']
report_df = pd.DataFrame(columns=cols)

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

#First basic models would be run
#Following that, a few would be shortlisted (on the basis of
#network architecture) and would be run with a grid

#NN1
dl_fit1 = H2ODeepLearningEstimator(model_id='dl_fit1',
                                   hidden=[50,30,20,5],
                                   seed=1,
                                   max_categorical_features=4,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
                                   adaptive_rate=True,
                                   )

dl_fit1.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward Grid"
#defined_params = str(nn_params1) + ": hidden=[50,20,30,5]"
defined_params = ""
file_name = "NN_1"
training_ratio = 0.65

os.getcwd()

fig_name = file_name + "_plot"
plot_info_1_base = plotMetricsWithThreshold(dl_fit1, test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True, model_type='h2o')

threshold = 0.3
report_df = report(dl_fit1, test, y , model_path, 4, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=threshold, sampling=None, model_type='basic')

#NN2
dl_fit2 = H2ODeepLearningEstimator(model_id='dl_fit2',
                                   hidden=[30,15],
                                   seed=1,
                                   max_categorical_features=4,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
                                   adaptive_rate=True
                                   )

dl_fit2.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward Grid"
#defined_params = str(nn_params1) + ": hidden=[50,20,30,5]"
defined_params = ""
file_name = "NN_2"
training_ratio = 0.65

os.getcwd()

fig_name = file_name + "_plot"
plot_info_2_base = plotMetricsWithThreshold(dl_fit2, test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True, model_type='h2o')

threshold = 0.41
report_df = report(dl_fit2, test, y , model_path, 4, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=threshold, sampling=None, model_type='basic')

#NN3
dl_fit3 = H2ODeepLearningEstimator(model_id='dl_fit3',
                                   hidden=[20,30,15],
                                   seed=1,
                                   max_categorical_features=4,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
                                   adaptive_rate=True
                                   )

dl_fit3.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward Grid"
#defined_params = str(nn_params1) + ": hidden=[50,20,30,5]"
defined_params = ""
file_name = "NN_3"
training_ratio = 0.65

os.getcwd()

fig_name = file_name + "_plot"
plot_info_2_base = plotMetricsWithThreshold(dl_fit3, test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True, model_type='h2o')

threshold = 0.27
report_df = report(dl_fit3, test, y , model_path, 4, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=threshold, sampling=None, model_type='basic')

#NN4
dl_fit4 = H2ODeepLearningEstimator(model_id='dl_fit4',
                                   hidden=[70,60,20,10],
                                   seed=1,
                                   max_categorical_features=4,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
                                   adaptive_rate=True
                                   )

dl_fit4.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward Grid"
#defined_params = str(nn_params1) + ": hidden=[50,20,30,5]"
defined_params = ""
file_name = "NN_4"
training_ratio = 0.65

os.getcwd()

fig_name = file_name + "_plot"
plot_info_4_base = plotMetricsWithThreshold(dl_fit4, test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True, model_type='h2o')

threshold = 0.7
report_df = report(dl_fit4, test, y , model_path, 4, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=threshold, sampling=None, model_type='basic')

#NN5
dl_fit5 = H2ODeepLearningEstimator(model_id='dl_fit5',
                                   hidden=[50,20,30,5],
                                   seed=1,
                                   max_categorical_features=4,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
                                   adaptive_rate=True
                                   )

dl_fit5.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward Grid"
#defined_params = str(nn_params1) + ": hidden=[50,20,30,5]"
defined_params = ""
file_name = "NN_5"
training_ratio = 0.46

os.getcwd()

fig_name = file_name + "_plot"
plot_info_5_base = plotMetricsWithThreshold(dl_fit5, test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True, model_type='h2o')

threshold = 0.46
report_df = report(dl_fit5, test, y , model_path, 4, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=threshold, sampling=None, model_type='basic')

#NN6
dl_fit6 = H2ODeepLearningEstimator(model_id='dl_fit6',
                                   hidden=[50,40,30,20,10],
                                   seed=1,
                                   max_categorical_features=4,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
                                   adaptive_rate=True
                                   )

dl_fit6.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward Grid"
#defined_params = str(nn_params1) + ": hidden=[50,20,30,5]"
defined_params = ""
file_name = "NN_6"
training_ratio = 0.46
num_layers = 5

os.getcwd()

fig_name = file_name + "_plot"
plot_info_5_base = plotMetricsWithThreshold(dl_fit6, test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True, model_type='h2o')

threshold = 0.59
report_df = report(dl_fit6, test, y , model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=threshold, sampling=None, model_type='basic')

#NN7
dl_fit7 = H2ODeepLearningEstimator(model_id='dl_fit7',
                                   hidden=[80,70,60,50,40,30,20,10],
                                   seed=1,
                                   max_categorical_features=4,
                                   export_weights_and_biases=True,
                                   shuffle_training_data=True,
                                   reproducible=True,
                                   adaptive_rate=True
                                   )

dl_fit7.train(x=x, y=y, training_frame=train)

model_name = "MLP - Feedforward Grid"
#defined_params = str(nn_params1) + ": hidden=[50,20,30,5]"
defined_params = ""
file_name = "NN_7"
training_ratio = 0.46
num_layers = 8

os.getcwd()

fig_name = file_name + "_plot"
plot_info_5_base = plotMetricsWithThreshold(dl_fit7, test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True, model_type='h2o')

threshold = 0.36
report_df = report(dl_fit7, test, y , model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=threshold, sampling=None, model_type='basic')

report_df.to_csv("Report_NN_V1_withClaim_predictRenewal_1.csv",index=False)

############################################################################
#NN1 - Grid
nn_params1 = {'rate_annealing': [0.7, 0.8, 0.9, 1.0],
                'epochs': [20, 30],
                'mini_batch_size' :[1, 10, 20, 30]
#                'rate' : [0.2, 0.3, 0.4],
                }

file_path = model_path + "\\dl_fit" + '2'
model_2 = h2o.load_model(file_path)

model_2.export_weights_and_biases = True
model_2.shuffle_training_data = True
model_2.reproducible = True
model_2.seed = 1

dl_grid_1_mod2 = H2OGridSearch(model=dl_fit1,
                          grid_id='dl_grid_1_mod2_mod2',
                          hyper_params=nn_params1)

dl_grid_1_mod2.train(x=x, y=y, training_frame=train)

nn_gridperf1_auc = dl_grid_1_mod2.get_grid(sort_by='auc', decreasing=True)
nn_gridperf1_accuracy = dl_grid_1_mod2.get_grid(sort_by='accuracy', decreasing=True)
nn_gridperf1_logloss = dl_grid_1_mod2.get_grid(sort_by='logloss', decreasing=True)
nn_gridperf1_logloss = dl_grid_1_mod2.get_grid(sort_by='recall', decreasing=True)


best_nn1 = nn_gridperf1.models[0]

# =============================================================================
# nn_gridperf1.as_data_frame()
# 
# nn_gridperf1_list = []
# for model_num in range(len(nn_gridperf1.models)):
#     nn_gridperf1_list.append(nn_gridperf1.models[model_num])
#     h2o.as_data_frame(nn_gridperf1.models[model_num])
# 
# table = nn_gridperf1.models[model_num]
# b = h2o.get_frame(table)
# 
# nn_gridperf1.models[model_num]
# 
# import sys
# old_target = sys.stdout
# f = open(model_path + "/" + "Try_1.txt", "w")
# sys.stdout = f
# table.show()
# sys.stdout = old_target
# f.close()
# =============================================================================
os.getcwd()
model_name = "MLP - Feedforward Grid"
defined_params = str(nn_params1) + ": hidden=[50,20,30,5]"
file_name = "NN_Grid_1"
training_ratio = 0.65

fig_name = file_name + "_plot"
plot_info_one = plotMetricsWithThreshold(best_nn1, test, y_test, fig_name=fig_name, fig_path = None,
                  precision="majority", recall="minority", 
                  fpoint5score = False, f1score=True, f2score=False, fbetascore=False,
                  TPR=True, FPR=True, specificity=True,
                  accuracy=True, misclassification=False,
                  display_info=True, model_type='h2o')

report_df = report(dl_fit1, test, y , model_path, 4, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=None, sampling=None, model_type='basic')
