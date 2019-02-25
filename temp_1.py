# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:51:16 2019

@author: sakshij
"""

import pandas as pd
import os

def returnColumnsList(list_of_cols):
    new_col_names = []
    for columns in list_of_cols:
        new_col_name = columns.split(".")[-1]
        new_col_names.append(new_col_name)
        
    return new_col_names

def changeColNamesAndSave(read_path, write_path, file_name):
    os.chdir(read_path)
    data_df = pd.read_csv(file_name)
    data_df.columns = returnColumnsList(data_df.columns.tolist())
    
    os.chdir(write_path)
    data_df.to_csv(file_name, index=False)

read_path = "D:\\Datasets\\CAS\\CASdatasets\\data_csv"
write_path = "D:\\Datasets\\CAS\\CASdatasets\\data_csv\\changed"

for root,dirs,files in os.walk(read_path):
    for file_name in files:
        print(file_name)
        if file_name.endswith(".csv"):
            changeColNamesAndSave(read_path, write_path, file_name)
        