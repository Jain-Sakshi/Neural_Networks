# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:16:59 2018

@author: sakshij
"""
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_score
import h2o
import sys
import joblib

# !conda install joblib

def report(model, test_data, target_coln, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold=None, sampling=None, model_type='basic', 
              store_model_perf = True, store_model=True, store_model_weights = True):
    
    if threshold == None:
        threshold = model.find_threshold_by_max_metric('f1')
    
    y_test = test_data[target_coln].as_data_frame()
    y_pred = predict_nn(model,test_data,threshold)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    try:
        TPR=(tp)/(tp+fn)
    except:
        TPR=-1
        
    try:
        FPR=(fp)/(fp+tn)
    except:
        FPR=-1
    
    clf_report = precision_recall_fscore_support(y_test, y_pred)
    
    accuracy = round(tp+tn)/len(y_test)
            
    training_ratio_str = str(training_ratio*100) + "%"
        
    if FPR != -1:
        specificity = 1 - FPR
    else:
        specificity = -1
        
    misclassification_rate = 1 - accuracy
    total_precision = precision_score(y_test,y_pred)
    features = test_data.shape[1] - 1
    
    try:
        best_par = model.best_params_
    except:
        best_par = ' '

    threshold = str(threshold)
    
    row1 = [training_ratio_str,features,sampling,model_name,tn,fp,threshold,'0',clf_report[0][0],clf_report[1][0],clf_report[2][0],clf_report[3][0],' ',accuracy,TPR,FPR,misclassification_rate,specificity,total_precision,file_name,defined_params]
    row2 = [' ',' ',' ',' ',fn,tp,' ','1',clf_report[0][1],clf_report[1][1],clf_report[2][1],clf_report[3][1],' ',' ',' ',' ',' ',' ',' ',' ',best_par]
    row3 = ['']
    
    model_report = pd.DataFrame([row1,row2,row3],columns=report_df.columns)
    report_df = report_df.append(model_report)
    report_df.index = list(range(len(report_df)))
    
    if store_model == True:
        h2o.save_model(model=model, path=model_path, force=True)
    
    if store_model_perf == True:
        #Saving model performance in a file
        model_perf = model.model_performance(test_data=test_data)
        old_target = sys.stdout
        f = open(model_path + "/" + file_name + "_Model_Perf.txt", "w")
        sys.stdout = f
        model_perf.show()
        sys.stdout = old_target
        f.close()

    if store_model_weights == True:
        #Saving weights and biases
        weights_list = []
        biases_list = []
        for layer_num in range(num_layers):
            weight_matrix = model.weights(matrix_id=0).as_data_frame()
            biases_vector = model.biases(vector_id=0).as_data_frame()
            weights_list.append(weight_matrix)
            biases_list.append(biases_vector)
        
        try:
            joblib.dump(weights_list,file_name + "Weights_Matrices_list.pkl")
        except:
            print("Weights not saved")
            
        try:
            joblib.dump(biases_list,file_name + "Biases_Vectors_list.pkl")
        except:
            print("Biases not saved")
    return report_df

def predict_nn(model,test_data,threshold,return_with_probability=False):
    y_pred_h2o_df = model.predict(test_data)
    y_pred_df = y_pred_h2o_df.as_data_frame()
    y_pred_proba = y_pred_df[['p1','p0']]
    
    predictions = []
    for index in y_pred_proba.index:
        if y_pred_proba.p1[index] > threshold:
            predictions.append(1)
        else:
            predictions.append(0)
            
    y_pred_proba['Prediction'] = predictions
    
    if return_with_probability == False:
        return y_pred_proba['Prediction']
    elif return_with_probability == True:
        return y_pred_proba
    else:
        print("Wrong value for parameter 'return_with_probability'")