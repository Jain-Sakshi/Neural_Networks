# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os

def createCriteriaFile(criteria_file_df, file_name, file_location = None):
    if file_location != None:
        os.chdir(file_location)
    criteria_file_df.to_csv(file_name + ".csv")

def addCriteria(metric, filter_value, class_,criteria_file_df):
    if metric == "fp_fn":
        if filter_value == True:
            filter_value = "FP > FN"
        elif filter_value == False:
            filter_value = "FP < FN"
    
    criteria_row = [metric, filter_value, class_]
    criteria_file_df.loc[len(criteria_file_df)] = criteria_row
    return criteria_file_df
    
def returnClassNumber(report_file, class_type):
    positives = report_file["True_Positives"][0]
    negatives = report_file["False_Positives"][0]
    
    if positives > negatives:
        if class_type == "majority":
            return '1'
        elif class_type == "minority":
            return '0'
        
    elif positives < negatives:
        if class_type == "majority":
            return '0'
        elif class_type == "minority":
            return '1'
    else:
        print("Both classes have equal counts")
        return -1

def filterByMetric(metric, report_file, criteria_file_df, filter_value, class_ = None):
    filtered_data = pd.DataFrame()
    
    if metric in ["accuracy", "TPR", "TNR"]:
        filtered_data = report_file[report_file[metric] >= filter_value]
        
    elif metric in ["fp_fn"]:
        if filter_value == True:
            filtered_data = report_file[report_file["False_Positives"] >= report_file["False_Negatives"]]
        elif filter_value == False:
            filtered_data = report_file[report_file["False_Positives"] <= report_file["False_Negatives"]]
        
    elif metric in ["FPR"]:
        filtered_data = report_file[report_file[metric] <= filter_value]
        
    elif metric in ["precision", "recall", "fpoint5score", "f1score", "f2score"]:
        if class_ in ["0", "1","majority", "minority"]:
            if class_ in ["majority", "minority"]:
                class_value = returnClassNumber(report_file, class_)
            else:
                class_value = class_
            
            filtered_data = report_file[report_file[metric + "_" + class_value] >= filter_value]    
        
        elif class_ in ["both"]:
            filter_class_0 = report_file[report_file[metric + "_0"] >= filter_value[0]]
            filter_class_1 = report_file[report_file[metric + "_1"] >= filter_value[1]]
            
            index_0 = filter_class_0.index
            index_1 = filter_class_1.index
            common_entries = index_1.intersection(index_0).tolist()
            
            filtered_data = filter_class_0.loc[common_entries]
    else:
        print("Incorrect metric passed for filtering")
        
    print("Filterd data is:")
    print(filtered_data)
    
    criteria_file_df = addCriteria(metric, filter_value, class_, criteria_file_df)
    
    return filtered_data


#model_report_file_name = "Model_2_0_all_plot_info"
#model_report_file_location = "D:\\Projects\\Tableau_Train_Combine\\NN\\Models\\V2_withClaim_predictRenewal_63\\grid\\model_2\\grid_all"

def getAccuracyNumber(file_name):
    accuracy_num = file_name[-1]
    return accuracy_num

#model_report_file_name = file_name
#model_report_file_location = models_path
#filtered_file_name = threshold_file_name
#filtered_file_location = threshold_files_path

def getIntersectionAndSave(all_filtered_dfs, filtered_file_name, filtered_file_location):
    filtered_data = all_filtered_dfs[0]
    for df_index in range(len(all_filtered_dfs) - 1):
        index_0 = all_filtered_dfs[df_index].index
        index_1 = all_filtered_dfs[df_index + 1].index
        common_entries = index_1.intersection(index_0).tolist()
        
        if len(common_entries) == 0:
            return None
        
        try:
            filtered_data = filtered_data.loc[common_entries]
        except:
            return
        
    filtered_data = filtered_data.drop_duplicates(keep=False)

    os.chdir(filtered_file_location)
    filtered_data.to_csv(filtered_file_name + ".csv", index=False)

def filterThresholds(model_report_file_name, model_report_file_location, 
                     criteria_num, filtered_file_name = None, 
                     filtered_file_location = None,
                     save_separate_files = True,
                     accuracy_filter = None, precision_filter = None, recall_filter = None, 
                     TPR_filter = None, FPR_filter = None, TNR_filter = None,
                     fpoint5score_filter = None, f1score_filter = None, f2score_filter = None,
                     precision_class = None, recall_class = None,
                     fpoint5score_class = None, f1score_class = None, f2score_class = None,
                     fp_less_than_fn = None, append_fp_less_than_fn = False
                     ):
    #filter requirement : filter values and class [majority, minority, 0, 1, both]
    #if both, _filter arg should be a list of 2 numbers
    #filter class requirement for metrics precision, recall, fscores, 
    
    os.chdir(model_report_file_location)
    model_report_file = pd.read_csv(model_report_file_name + ".csv")
    
    criteria_file_df = pd.DataFrame(columns=["Metric","Filter_Value","Class"])
    
    if filtered_file_name == None:
        filtered_file_name = model_report_file_name + "_threshold"
    if filtered_file_name.endswith("threshold") == False:
        filtered_file_name = filtered_file_name + "_threshold"
        
    if filtered_file_location == None:
        filtered_file_location = model_report_file_location + "\\Threshold\\Criteria_" + str(criteria_num)
        
    if os.path.exists(filtered_file_location):
        print("Path exists")
    else:
        os.makedirs(filtered_file_location)
        print("Path did not exist: Path created")
    
    if save_separate_files == True:
        accuracy_num = getAccuracyNumber(model_report_file_name)
        separate_files_location = filtered_file_location + "\\Accuracy_" + accuracy_num
        os.chdir(separate_files_location)
    
    all_filtered_dfs = []
    
    if accuracy_filter != None:
        filter_accuracy = filterByMetric("accuracy", model_report_file, criteria_file_df, accuracy_filter)
        all_filtered_dfs.append(filter_accuracy)
        if save_separate_files == True:
             filter_accuracy.to_csv(filtered_file_name + "_accuracy_filter.csv",index=False)
    
    if precision_class != None:
        filter_precision = filterByMetric("precision",model_report_file, criteria_file_df, precision_filter, precision_class)
        all_filtered_dfs.append(filter_precision)
        if save_separate_files == True:
             filter_precision.to_csv(filtered_file_name + "_precision_filter.csv",index=False)
        
    if recall_class != None:
        filter_recall = filterByMetric("recall", model_report_file, criteria_file_df, recall_filter, recall_class)
        all_filtered_dfs.append(filter_recall)
        if save_separate_files == True:
             filter_recall.to_csv(filtered_file_name + "_recall_filter.csv",index=False)
        
    if TPR_filter != None:
        filter_TPR = filterByMetric("TPR", model_report_file, criteria_file_df, TPR_filter)
        all_filtered_dfs.append(filter_TPR)
        if save_separate_files == True:
             filter_TPR.to_csv(filtered_file_name + "_TPR_filter.csv",index=False)
        
    if FPR_filter != None:
        filter_FPR = filterByMetric("FPR", model_report_file, criteria_file_df, FPR_filter)
        all_filtered_dfs.append(filter_FPR)
        if save_separate_files == True:
             filter_FPR.to_csv(filtered_file_name + "_FPR_filter.csv",index=False)
        
    if TNR_filter != None:
        filter_TNR = filterByMetric("TNR", model_report_file, criteria_file_df, TNR_filter)
        all_filtered_dfs.append(filter_TNR)
        if save_separate_files == True:
             filter_TNR.to_csv(filtered_file_name + "_TNR_filter.csv",index=False)
        
    if fpoint5score_class != None:
        filter_fpoint5 = filterByMetric("fpoint5score", model_report_file, criteria_file_df, fpoint5score_filter, fpoint5score_class)
        all_filtered_dfs.append(filter_fpoint5)
        if save_separate_files == True:
             filter_fpoint5.to_csv(filtered_file_name + "_fpoint5score_filter.csv",index=False)
        
    if f1score_class != None:
        filter_f1 = filterByMetric("f1score", model_report_file, criteria_file_df, f1score_filter, f1score_class)
        all_filtered_dfs.append(filter_f1)
        if save_separate_files == True:
             filter_f1.to_csv(filtered_file_name + "_f1score_filter.csv",index=False)
        
    if f2score_class != None:
        filter_f2 = filterByMetric("f2score", model_report_file, criteria_file_df, f2score_filter, f2score_class)
        all_filtered_dfs.append(filter_f2)
        if save_separate_files == True:
             filter_f2.to_csv(filtered_file_name + "_f2score_filter.csv",index=False)
        
    if fp_less_than_fn != None:
        filter_fp_fn = filterByMetric("fp_fn", model_report_file, criteria_file_df, fp_less_than_fn)
        if append_fp_less_than_fn == True:
            all_filtered_dfs.append(filter_fp_fn)
        if save_separate_files == True:
             filter_fp_fn.to_csv(filtered_file_name + "_fp_fn.csv",index=False)
        
       
    createCriteriaFile(criteria_file_df, filtered_file_name + "_info")   
    getIntersectionAndSave(all_filtered_dfs, filtered_file_name, filtered_file_location)
    
def getModelNum(file_name):
    name_split = file_name.split('_')
    model_num = name_split[2]
    
    return model_num

#def getArchNum(file_name):
#    name_split = file_name.split('_')
#    model_num = name_split[1]
#    
#    return model_num
    
def appendThresholdFiles(report_file, model_num, arch_num, report_threshold_df):
    temp_report_file = report_file
    temp_report_file["Model_Num"] = model_num
    temp_report_file["Arch_Num"] = arch_num
    
    report_threshold_df = report_threshold_df.append(temp_report_file)
    
    return report_threshold_df

def combineModelReports(use_case_models_location, grid_or_pretrained,
                        combined_file_name, combined_file_location,
                        no_of_archs):
#    assumption : nomanclature of files is of the form:
#                   Model_<arch_num>_<model_num>_all_plot_info.csv
    
    report_threshold_df = pd.DataFrame()
    
    for arch_num in range(1,no_of_archs+1):
        models_path = use_case_models_location + "\\" + grid_or_pretrained + "\\model_" + str(arch_num) + "\\grid_selected"
    
        for root,dirs,files in os.walk(models_path):
            for file in files:
               if file.endswith(".csv"):
                   report_file = pd.read_csv(file)
                   if len(report_file) != 0:
                       model_num = getModelNum(file)
                       report_threshold_df = appendThresholdFiles(report_file, model_num, arch_num, report_threshold_df)
                   
    if combined_file_location == None:
        combined_file_location = use_case_models_location + "\\" + grid_or_pretrained
    
    os.chdir(combined_file_location)
    
    if len(report_threshold_df) != 0:
        report_threshold_df.to_csv(combined_file_name + ".csv")
    else:
        print("No thresholds of any architecture satisfy the given criteria")

def getArchNum(file_name):
    file_name_split = file_name.split("_")
    arch_num = file_name_split[1]
    return arch_num

def getModelNum(file_name):
    file_name_split = file_name.split("_")
    model_num = file_name_split[2]
    return model_num


def appendAllUseCaseFiles(grid_model_path, no_of_models, all_data_file_name):
    all_data_file = pd.DataFrame()
    
    shortlisted_model_folders = []
    for index in range(1,no_of_models + 1):
        folder_name = grid_model_path + "\\model_" + str(index) + "\\grid_selected"
        shortlisted_model_folders.append(folder_name)
    
    for folder in shortlisted_model_folders:
        for root,dirs,files in os.walk(folder):
            for file_name in files:
                print(file_name)
                if file_name.startswith("Model_") and file_name.endswith(".csv") and "threshold" not in file_name:
                    print(file_name + " is shortlisted")
                    report_file = pd.read_csv(root + "\\" + file_name)
                    report_file["Arch_Num"] = getArchNum(file_name)
                    report_file["Model_Num"] = getModelNum(file_name)
                    all_data_file = all_data_file.append(report_file)
                   
    os.chdir(grid_model_path)
    all_data_file.to_csv(all_data_file_name + ".csv", index = False)
    return all_data_file