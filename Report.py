# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:16:59 2018

@author: sakshij
"""
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_score
import joblib

def report(model,model_name,X_test,y_test,split_ratio,sampling,model_filename,report_df,parameters=None, model_threshold = None, save_file = True):
    y_pred=model.predict(X_test)
    
    if model_threshold != None:
        temp_y_pred = []
        for prediction in y_pred:
            if prediction >= model_threshold:
                temp_y_pred.append(1)
            else:
                temp_y_pred.append(0)
        y_pred = temp_y_pred
        model_threshold = str(model_threshold)
    else:
        model_threshold = ' '
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    TPR=(tp)/(tp+fn)
    FPR=(fp)/(fp+tn)
    clf_report = precision_recall_fscore_support(y_test, y_pred)
    
    try:
        accuracy = model.score(X_test,y_test)
    except:
        accuracy = round(tp+tn)/len(y_test)
        
    try:
        best_par = model.best_params_
    except:
        best_par = ' '
    
    try:
        split_ratio = float(split_ratio)
        split_ratio_str = str(100-(split_ratio*100)) + '-' + str(split_ratio*100)
    except:
        split_ratio_str = split_ratio
        
    misclassification_rate = 1 - accuracy
    specificity = 1 - FPR
    total_precision = precision_score(y_test,y_pred, average = "weighed")
    features = X_test.shape[1]
    
    param = []
    if parameters == None:
        param = 'basic'
    elif parameters == 'grid':
        param = model.param_grid
    else:
        param = parameters
    
    row1 = [split_ratio_str,features,sampling,model_name,tn,fp,model_threshold,'0',clf_report[0][0],clf_report[1][0],clf_report[2][0],clf_report[3][0],' ',accuracy,TPR,FPR,misclassification_rate,specificity,total_precision,model_filename,param]
    row2 = [' ',' ',' ',' ',fn,tp,' ','1',clf_report[0][1],clf_report[1][1],clf_report[2][1],clf_report[3][1],' ',' ',' ',' ',' ',' ',' ',' ',best_par]
    row3 = ['']
    
    dat = pd.DataFrame([row1,row2,row3],columns=report_df.columns)
    report_df = report_df.append(dat)
    
    if save_file == True:
        joblib.dump(model, model_filename + ".pkl")
    return report_df

