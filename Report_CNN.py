# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:16:59 2018

@author: sakshij
"""
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_score
from tensorflow import keras
import sys

# !conda install joblib

def report(model, X_test, y_test, model_path, num_layers, report_df, 
              model_name, defined_params, file_name, training_ratio, 
              threshold, sampling=None, model_type='basic'):
    
    y_pred_score = model.predict(X_test)
    
    y_pred = []
    for prediction in y_pred_score:
        if prediction > threshold:
            y_pred.append(1)
        elif prediction < threshold:
            y_pred.append(0)
    
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
    features = X_test.shape[1]
    
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

    #Saving model and weights
    model_json = model.to_json()
    with open(file_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(file_name + ".h5")
    print("Saved model to disk")
    
    #Saving model info in a file
    old_target = sys.stdout
    f = open(model_path + "/" + file_name + "_Model_Info.txt", "w")
    sys.stdout = f
    model.summary()
    sys.stdout = old_target
    f.close()
    
    return report_df

def saveModelandModelInfo(model, file_name, file_path):
    #Saving model and weights
    model_json = model.to_json()
    with open(file_name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(file_name + ".h5")
    print("Saved model to disk")
    
    #Saving model info in a file
    old_target = sys.stdout
    f = open(file_path + "/" + file_name + "_Model_Info.txt", "w")
    sys.stdout = f
    model.summary()
    sys.stdout = old_target
    f.close()