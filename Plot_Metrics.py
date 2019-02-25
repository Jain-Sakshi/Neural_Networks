# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 13:52:27 2018

@author: sakshij
"""

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, fbeta_score
import numpy as np
import pandas as pd
import os
from scipy.stats import chisquare

def calculatePrecision(y_test, y_pred, threshold_range):
    precision_0 = []
    precision_1 = []
    for threshold in threshold_range:
        y_pred_threshold = (y_pred > threshold).astype('int')
        precision_0.append(precision_score(y_test, y_pred_threshold,pos_label=0))
        precision_1.append(precision_score(y_test, y_pred_threshold,pos_label=1))
    
    precision_dict = {0: precision_0, 1: precision_1}
    return precision_dict

def calculateRecall(y_test, y_pred, threshold_range):
    recall_0 = []
    recall_1 = []
    for threshold in threshold_range:
        y_pred_threshold = (y_pred > threshold).astype('int')
        recall_0.append(recall_score(y_test, y_pred_threshold,pos_label=0))
        recall_1.append(recall_score(y_test, y_pred_threshold,pos_label=1))
    
    recall_dict = {0: recall_0, 1: recall_1}
    return recall_dict

def calculateFScore(y_test, y_pred, threshold_range,beta):
    fscore_0 = []
    fscore_1 = []
    for threshold in threshold_range:
        y_pred_threshold = (y_pred > threshold).astype('int')
        fbeta_score_0 = fbeta_score(y_test, y_pred_threshold, beta=beta, average='binary', pos_label=0)
        fbeta_score_1 = fbeta_score(y_test, y_pred_threshold, beta=beta, average='binary', pos_label=1)
        fscore_0.append(fbeta_score_0)
        fscore_1.append(fbeta_score_1)
    
    fscore_dict = {0: fscore_0, 1: fscore_1}
    return fscore_dict

def calculateAccuracy(y_test, y_pred, threshold_range):
    accuracy = []
    for threshold in threshold_range:
        y_pred_threshold = (y_pred > threshold).astype('int')
        accuracy.append(accuracy_score(y_test, y_pred_threshold))
    
    return accuracy

def calculateMisclassification(y_test, y_pred, threshold_range):
    accuracy = calculateAccuracy(y_test, y_pred, threshold_range)
    misclassification = [1 - x for x in accuracy]
    
    return misclassification

def calculateTPR(y_test, y_pred, threshold_range):
    TPR_all = []
    for threshold in threshold_range:
        y_pred_threshold = (y_pred > threshold).astype('int')
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
        try:
            TPR=(tp)/(tp+fn)
        except:
            TPR=-0.2
        TPR_all.append(TPR)
    
    return TPR_all

def calculateFPR(y_test, y_pred, threshold_range):
    FPR_all = []
    for threshold in threshold_range:
        y_pred_threshold = (y_pred > threshold).astype('int')
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
        try:
            FPR=(fp)/(fp+tn)
        except:
            FPR=-0.2
        FPR_all.append(FPR)
    
    return FPR_all

def calculateTNR(y_test, y_pred, threshold_range):
    FPR_all = calculateFPR(y_test, y_pred, threshold_range)
    TNR_all = [1 - x for x in FPR_all]
    
    return TNR_all

def calculateChiSquare(y_test, y_pred, threshold_range):
    observed_counts = pd.value_counts(y_test)
    chi_statistics = []
    p_values = []
    for threshold in threshold_range:
        y_pred_threshold = (y_pred > threshold).astype('int')
        
        expected_counts = pd.value_counts(y_pred_threshold)

        chi_statistic, p_value = chisquare(observed_counts.values, expected_counts.values)
        chi_statistics.append(chi_statistic)
        p_values.append(p_value)
    
    return chi_statistics, p_values

def calculateYoudensIndex(y_test, y_pred, threshold_range):
    TNR = calculateTNR(y_test, y_pred, threshold_range)
    TPR = calculateTPR(y_test, y_pred, threshold_range)
    
    youdens_index = []
    for index in range(len(TNR)):
        youdens_index.append(TNR[index] + TPR[index] - 1)
        
    return youdens_index

def calculateConfusionMatrix(y_test, y_pred, threshold_range):
    tn_all = []
    fp_all = []
    fn_all = []
    tp_all = []    
    for threshold in threshold_range:
        y_pred_threshold = (y_pred > threshold).astype('int')
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_threshold).ravel()
        tn_all.append(tn)
        fp_all.append(fp)
        fn_all.append(fn)
        tp_all.append(tp)
    
    return [tn_all, fp_all, fn_all, tp_all]

def appendBasedOnParams(metric, metric_info, metric_parameter_value, majority_class, minority_class, counts, plot_info, plot_labels, threshold_range):
    if metric in ['precision','recall']:
        if metric_parameter_value == 'both':
            plot_info.append(metric_info[0])
            plot_info.append(metric_info[1])
            plot_labels.append(metric + "_0")
            plot_labels.append(metric + "_1")
            plt.plot(threshold_range, metric_info[0])
            plt.plot(threshold_range, metric_info[1])
            
        elif metric_parameter_value == '0':
            plot_info.append(metric_info[0])
            plot_labels.append(metric + "_0")
            plt.plot(threshold_range, metric_info[0])
            
        elif metric_parameter_value == '1':
            plot_info.append(metric_info[1])
            plot_labels.append(metric + "_1")
            plt.plot(threshold_range, metric_info[1])
        
        elif metric_parameter_value == 'total' or metric_parameter_value == 'weighed':
            total_values = (counts[0]*metric_info[0]) + (counts[1]*metric_info[1])
            plot_info.append(total_values)
            plot_labels.append(metric + "_Total")
            plt.plot(threshold_range, total_values)
        
        elif metric_parameter_value == 'inverse_weighed':
            total_counts = counts[0] + counts[1]
            total_values = ((total_counts/counts[0])*metric_info[0]) + ((total_counts/counts[1])*metric_info[1])
            plot_info.append(total_values)
            plot_labels.append(metric + "_Total")
            plt.plot(threshold_range, total_values)
        
        elif metric_parameter_value == 'majority':
            plot_info.append(metric_info[majority_class])
            plot_labels.append(metric + "_" + str(majority_class))
            plt.plot(threshold_range, metric_info[majority_class])
            
        elif metric_parameter_value == 'minority':
            plot_info.append(metric_info[minority_class])
            plot_labels.append(metric + "_" + str(minority_class))
            plt.plot(threshold_range, metric_info[minority_class])
        else:
            raise Exception("No category")
    
    elif metric in ['fpoint5score','f1score','f2score']:
        plot_info.append(metric_info[0])
        plot_info.append(metric_info[1])
        plot_labels.append(metric + "_0")
        plot_labels.append(metric + "_1")
        plt.plot(threshold_range, metric_info[0])
        plt.plot(threshold_range, metric_info[1])
        
    elif metric in ['fbetascore']:
        plot_info.append(metric_info[0])
        plot_info.append(metric_info[1])
        plot_labels.append('f' + str(metric_parameter_value) + 'score' + "_0")
        plot_labels.append('f' + str(metric_parameter_value) + 'score' + "_1")
        plt.plot(threshold_range, metric_info[0])
        plt.plot(threshold_range, metric_info[1])
    
    elif metric in ['TPR','FPR','TNR','accuracy','misclassification-rate']:
        plot_info.append(metric_info)
        plot_labels.append(metric)
        plt.plot(threshold_range, metric_info)
    
    return plot_info, plot_labels 
        
def plotAndSave(metric, metric_info, fig_name, threshold_range):
    if metric == "chi_square":
        metric_info_chi_statistic = metric_info[0]
        metric_info_p_value = metric_info[1]
        
        plt.plot(threshold_range, metric_info_chi_statistic)
        plt.title('Chi statistic with Threshold')
        plt.legend(["Chi statistic"])
        plt.xlabel('Threshold')
        plt.ylabel('Chi statistic')
        plt.show()
        plt.savefig(fig_name + "_chi_statistic.png")
        plt.close()
        
        plt.plot(threshold_range, metric_info_p_value)
        plt.title('p value with Threshold')
        plt.legend(["p value"])
        plt.xlabel('Threshold')
        plt.ylabel('p value')
        plt.show()
        plt.savefig(fig_name + "_p_value.png")
        plt.close()
        
    elif metric == "youdens_index":
        plt.plot(threshold_range, metric_info)
        plt.title('Youdens Index with Threshold')
        plt.legend(["Youdens Index"])
        plt.xlabel('Threshold')
        plt.ylabel('Youdens Index')
        plt.show()
        plt.savefig(fig_name + "_Youdens_index.png")
        plt.close()
    
    elif metric == "confusion_matrix":
        tn_all = metric_info[0]
        fp_all = metric_info[1]
        fn_all = metric_info[2]
        tp_all = metric_info[3]
        
        plt.plot(threshold_range, tn_all)
        plt.plot(threshold_range, fp_all)
        plt.plot(threshold_range, fn_all)
        plt.plot(threshold_range, tp_all)
        plt.title('Confusion Matrix')
        plt.legend(["TN","FP","FN","TP"])
        plt.xlabel('Threshold')
        plt.ylabel('No of rows')
        plt.show()
        plt.savefig(fig_name + "_Confusion_Matrix.png")
        plt.close()

def createRequiredData(plot_labels, all_plot_info, threshold_range):
    plot_info_df = pd.DataFrame()
    plot_info_df["Threshold"] = threshold_range
    
    plot_info_df['chi_statistic'] = all_plot_info['chi_statistic']
    plot_info_df['p_value'] = all_plot_info['p_value']
    plot_info_df['youdens_index'] = all_plot_info['youdens_index']
    
    plot_info_df['True_Negatives'] = all_plot_info['True_Negatives']
    plot_info_df['False_Positives'] = all_plot_info['False_Positives']
    plot_info_df['False_Negatives'] = all_plot_info['False_Negatives']
    plot_info_df['True_Positives'] = all_plot_info['True_Positives']
    
    for label in plot_labels:
        info = all_plot_info[label]
        plot_info_df[label] = info
        
    return plot_info_df

def addData(metric, metric_info, all_plot_info):
    if metric in ['precision','recall','fpoint5score','f1score','f2score','fbetascore']:
        all_plot_info[metric + '_0'] = metric_info[0]
        all_plot_info[metric + '_1'] = metric_info[1]
    elif metric in ['TPR','FPR','TNR','accuracy','misclassification-rate','youdens_index']:
        all_plot_info[metric] = metric_info
    elif metric in ['chi_square']:
        print(len(metric))
        metric_info_chi_statistic = metric_info[0]
        metric_info_p_value = metric_info[1]
        all_plot_info['chi_statistic'] = metric_info_chi_statistic
        all_plot_info['p_value'] = metric_info_p_value
    elif metric in ['confusion_matrix']:
        tn_all = metric_info[0]
        fp_all = metric_info[1]
        fn_all = metric_info[2]
        tp_all = metric_info[3]
        
        all_plot_info['True_Negatives'] = tn_all
        all_plot_info['False_Positives'] = fp_all
        all_plot_info['False_Negatives'] = fn_all
        all_plot_info['True_Positives'] = tp_all
    else:
        raise Exception("Incorrect Metric")
    
    return all_plot_info

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
    elif return_with_probability == '1':
        return y_pred_proba['p1']
    elif return_with_probability == '0':
        return y_pred_proba['p0']
    else:
        print("Wrong value for parameter 'return_with_probability'")

def plotMetricsWithThreshold(model, X_test, y_test, fig_name=None, fig_path = None,
                             precision=False, recall=False, 
                             fpoint5score = False, f1score=False, f2score=False, fbetascore=False,
                             TPR=False, FPR=False, specificity=False,
                             accuracy=False, misclassification=False,
                             chi_square=True, youdens_index=True, confusion_matrix=True,
                             display_info=False,model_type=None,
                             return_max_metric=False,
                             return_max_metric_threshold=False):
    
    plt.gcf().clear()
    if model_type == 'h2o':
        threshold = 0.5
        test_data = X_test
        y_pred = predict_nn(model,test_data,threshold,return_with_probability='1')
        
    else:
        y_pred = model.predict(X_test)
        y_pred = y_pred.reshape(1,len(y_pred))
        y_pred = y_pred[0]
        
    majority = pd.value_counts(y_test).idxmax()
    minority = (majority == 0).astype(int)
    majority = int(majority)

    counts = pd.value_counts(y_test)
    
    threshold_range = np.arange(0,1,0.01)
    
    plot_labels = []
    plot_info = []
    
    all_plot_info = pd.DataFrame()
    all_plot_info["Threshold"] = threshold_range
    
    precision_recall_params = ["both","0","1","total","majority","minority",False]
    
    confusion_matrix_info = calculateConfusionMatrix(y_test, y_pred, threshold_range)
    all_plot_info = addData('confusion_matrix', confusion_matrix_info, all_plot_info)
    if confusion_matrix is not False and display_info is not False:
        plotAndSave('confusion_matrix',confusion_matrix_info, fig_name, threshold_range)
    
    chi_statistic_info, p_value_info = calculateChiSquare(y_test, y_pred, threshold_range)
    all_plot_info = addData('chi_square', [chi_statistic_info, p_value_info], all_plot_info)
    if chi_square is not False and display_info is not False:
        plotAndSave('chi_square',[chi_statistic_info, p_value_info], fig_name, threshold_range)
        
    youdens_index_info = calculateYoudensIndex(y_test, y_pred, threshold_range)
    all_plot_info = addData('youdens_index', youdens_index_info, all_plot_info)
    if youdens_index is not False and display_info is not False:
        plotAndSave('youdens_index',youdens_index_info, fig_name, threshold_range)
    
    precision_info = calculatePrecision(y_test, y_pred, threshold_range)    
    all_plot_info = addData('precision', precision_info, all_plot_info)
    if precision not in precision_recall_params:
        raise ValueError("Wrong value! Enter from " + precision_recall_params)
    else:
        if precision is not False and display_info is not False:
            plot_info, plot_labels = appendBasedOnParams('precision', precision_info, precision, majority, minority, counts, plot_info, plot_labels,threshold_range)
            
    recall_info = calculateRecall(y_test, y_pred,threshold_range)
    all_plot_info = addData('recall', recall_info, all_plot_info)
    if recall not in precision_recall_params:
        raise ValueError("Wrong value! Enter from " + precision_recall_params)
    else:
        if recall is not False and display_info is not False:
            plot_info, plot_labels = appendBasedOnParams('recall', recall_info, recall, majority, minority, counts, plot_info, plot_labels,threshold_range)

    fpoint5score_info = calculateFScore(y_test, y_pred, threshold_range, 0.5)
    all_plot_info = addData('fpoint5score', fpoint5score_info, all_plot_info)
    if fpoint5score is not False and display_info is not False:
        plot_info, plot_labels = appendBasedOnParams('fpoint5score', fpoint5score_info, fpoint5score, majority, minority, counts, plot_info, plot_labels,threshold_range)

    f1score_info = calculateFScore(y_test, y_pred, threshold_range,1)        
    all_plot_info = addData('f1score', f1score_info, all_plot_info)
    if f1score is not False and display_info is not False:
        plot_info, plot_labels = appendBasedOnParams('f1score', f1score_info, f1score, majority, minority, counts, plot_info, plot_labels,threshold_range)
    
    f2score_info = calculateFScore(y_test, y_pred, threshold_range,2)        
    all_plot_info = addData('f2score', f2score_info, all_plot_info)
    if f2score is not False and display_info is not False:
        plot_info, plot_labels = appendBasedOnParams('f2score', f2score_info, f2score, majority, minority, counts, plot_info, plot_labels,threshold_range)

    if fbetascore is not False and display_info is not False:
        fbetascore_info = calculateFScore(y_test, y_pred, threshold_range,fbetascore)        
        all_plot_info = addData('fbetascore', fbetascore_info, all_plot_info)
        plot_info, plot_labels = appendBasedOnParams('fbetascore', fbetascore_info, fbetascore, majority, minority, counts, plot_info, plot_labels, threshold_range)

    TPR_info = calculateTPR(y_test, y_pred, threshold_range)        
    all_plot_info = addData('TPR', TPR_info, all_plot_info)
    if TPR is not False and display_info is not False:
        plot_info, plot_labels = appendBasedOnParams('TPR', TPR_info, TPR, majority, minority, counts, plot_info, plot_labels, threshold_range)

    FPR_info = calculateFPR(y_test, y_pred, threshold_range)
    all_plot_info = addData('FPR', FPR_info, all_plot_info)
    if FPR is not False and display_info is not False:
        plot_info, plot_labels = appendBasedOnParams('FPR', FPR_info, FPR, majority, minority, counts, plot_info, plot_labels, threshold_range)

    specificity_info = calculateTNR(y_test, y_pred, threshold_range)        
    all_plot_info = addData('TNR', specificity_info, all_plot_info)
    if specificity is not False and display_info is not False:
        plot_info, plot_labels = appendBasedOnParams('TNR', specificity_info, specificity, majority, minority, counts, plot_info, plot_labels, threshold_range)

    accuracy_info = calculateAccuracy(y_test, y_pred, threshold_range)        
    all_plot_info = addData('accuracy', accuracy_info, all_plot_info)
    if accuracy is not False and display_info is not False:
        plot_info, plot_labels = appendBasedOnParams('accuracy', accuracy_info, accuracy, majority, minority, counts, plot_info, plot_labels, threshold_range)

    misclassification_info = calculateMisclassification(y_test, y_pred, threshold_range)        
    all_plot_info = addData('misclassification-rate', misclassification_info, all_plot_info)
    if misclassification is not False and display_info is not False:
        plot_info, plot_labels = appendBasedOnParams('misclassification', misclassification_info, misclassification, majority, minority, counts, plot_info, plot_labels, threshold_range)
    
    if fig_path is not None:
        try:
            os.chdir(fig_path)
        except:
            raise Exception("Path not found.")
    
    
    if display_info is not False:
        plt.title('Metrics changing with Threshold')
        plt.legend(plot_labels)
        plt.xlabel('Threshold')
        plt.ylabel('Metric Range')
        plt.savefig(fig_name + ".png")
        
        plot_info_df = createRequiredData(plot_labels, all_plot_info, threshold_range)
    
    all_plot_info.to_csv(fig_name + "_all_plot_info.csv",index=False)
    
    return_info = []
    
    if display_info is not False:
        return_info.append(plot_info_df)
    if return_max_metric is not False:
        if return_max_metric == 'accuracy':
            max_accuracy = max(all_plot_info['accuracy'])
            return_info.append(max_accuracy)
    
    if len(return_info) == 1:
        return return_info[0]
    else:
        return return_info
    
def plotMetricsWithReport(report_file_name, report_file_location):
    return 1