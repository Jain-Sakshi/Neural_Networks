# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 14:45:12 2019

@author: sakshij
"""
import pandas as pd
import os

def replaceSpecialChars(string):
    special_chars_list = list("-*.()")
    replace_char = "-"
    for element in string:
        if element in special_chars_list:
            string = string.replace(element,replace_char)
    return string

path = "D:\\Datasets\\Traffic_Violations_Maryland_County"
file_name = "Traffic_Violations"

os.chdir(path)

full_data = pd.read_csv(file_name + ".csv")

list_of_cols = ["Description","Charge"]
data_df = full_data[list_of_cols]
data_df["Charge_Change_1"] = data_df["Charge"].apply(lambda x : replaceSpecialChars(x))
data_df["Charge_Num"] = data_df["Charge_Change_1"].apply(lambda x : x.split('-')[0])

charge_num_vc = pd.value_counts(data_df["Charge_Num"])

data_df["Charge_Num_Seq"] = data_df["Charge_Change_1"].apply(lambda x : "-".join(x.split('-')[:2]))

charge_seq_vc = pd.value_counts(data_df["Charge_Num_Seq"])
#take top 16 : via 2% threshold

