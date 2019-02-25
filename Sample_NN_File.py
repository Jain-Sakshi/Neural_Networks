# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 10:18:04 2018

@author: sakshij
"""

import pandas as pd

import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import os
import pandas as pd
from sklearn.model_selection import train_test_split

############################################################################
#Paths
model_path = "D:\\Projects\\Tableau_Train_Combine\\V2_Claim_noclaim"
data_file_path = "D:\\Projects\\Tableau_Train_Combine\\Data_Split_for_Model_Training"
data_file = "Cleansed_Encoded_Vehicle2_Latest"

os.getcwd()
#os.chdir('D:\\Data_Nik\\Vehicle1_Encoder_H')

############################################################################
#Data        

############################################################################
#Modelling
