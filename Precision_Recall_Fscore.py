# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:30:39 2018

@author: sakshij
"""


from collections import Counter

import pandas as pd
import string
import nltk
import re
from nltk.corpus import wordnet as wn

from nltk import ngrams
from itertools import chain
from collections import Counter

def flatten(two_dim_list):
    return list(chain.from_iterable(two_dim_list))

def getAllGrams(data_df,data_coln,n):
    data_split = data_df[data_coln].apply(lambda x : x.split(' '))
    
    grams_all_doc = []
    for doc in data_split:
        grams = list(ngrams(doc,n))
        grams_all_doc.append(grams)
        
    grams_all_doc_flattened = flatten(grams_all_doc)
    vc_all_grams = pd.value_counts(grams_all_doc_flattened)
    gram_vocab_df = pd.DataFrame()
    gram_vocab_df['7_Grams'] = list(set(vc_all_grams.index))
    gram_vocab_df['Frequency'] = vc_all_grams.values
    return gram_vocab_df

def getNGramsInfo(data_df,data_coln,target_coln,index_coln,n,threshold=50):
    data_split = data_df[data_coln].apply(lambda x : x.split(' '))
    
    grams_all_doc = []
    grams_doc_dict = {}
    for doc_index in data_df.index:
        document = data_split[doc_index]
        grams = list(ngrams(document,n))
        grams_all_doc.append(grams)
        for gram in grams:
            try:
                grams_doc_dict[gram].append(doc_index)
            except:
                grams_doc_dict[gram] = []
                grams_doc_dict[gram].append(doc_index)
    
    gram_vocab_df_all = pd.DataFrame()
    gram_vocab_df_all['NGram'] = grams_doc_dict.keys()
    gram_vocab_df_all['Doc_Set'] = grams_doc_dict.values()
    gram_vocab_df_all['Doc_Length'] = gram_vocab_df_all['Doc_Set'].apply(lambda x : len(x))
    
    gram_vocab_df = gram_vocab_df_all[gram_vocab_df_all['Doc_Length'] >= threshold]
    
    print("Calculating Metrics for 1")
    
    all_1_docs = data_df[index_coln][data_df[target_coln] == 1.0]
    len_all_1_docs = len(all_1_docs)
    
    precision_1 = []
    recall_1 = []
    f1_score_1 = []
    
    for word in gram_vocab_df['NGram']:
        docs_with_phrase_series = gram_vocab_df[gram_vocab_df.NGram == word]
        docs_with_phrase = list(docs_with_phrase_series.Doc_Set)[0]
        phrase_nonzero = set(docs_with_phrase)
        r1_with_phrase = phrase_nonzero.intersection(all_1_docs)
        precision_of_doc = len(r1_with_phrase)/len(phrase_nonzero)
        recall_of_doc = len(r1_with_phrase)/len_all_1_docs
        try:
            f1_score_of_doc = (2*precision_of_doc*recall_of_doc)/(2*(precision_of_doc+recall_of_doc))
        except:
            f1_score_of_doc = -1
        
        precision_1.append(precision_of_doc)
        recall_1.append(recall_of_doc)
        f1_score_1.append(f1_score_of_doc)
    
    gram_vocab_df['Precision_1'] = precision_1
    gram_vocab_df['Recall_1'] = recall_1
    gram_vocab_df['F1_score_1'] = f1_score_1
    
    print("Calculating Metrics for 0")
    
    all_0_docs = data_df[index_coln][data_df[target_coln] == 0.0]
    len_all_0_docs = len(all_0_docs)
    
    precision_0 = []
    recall_0 = []
    f1_score_0 = []
    
    for word in gram_vocab_df['NGram']:
        docs_with_phrase_series = gram_vocab_df[gram_vocab_df.NGram == word]
        docs_with_phrase = list(docs_with_phrase_series.Doc_Set)[0]
        phrase_nonzero = set(docs_with_phrase)
        r0_with_phrase = phrase_nonzero.intersection(all_0_docs)
        precision_of_doc = len(r0_with_phrase)/len(phrase_nonzero)
        recall_of_doc = len(r0_with_phrase)/len_all_0_docs
        try:
            f0_score_of_doc = (2*precision_of_doc*recall_of_doc)/(2*(precision_of_doc+recall_of_doc))
        except:
            f0_score_of_doc = -1
        
        precision_0.append(precision_of_doc)
        recall_0.append(recall_of_doc)
        f1_score_0.append(f0_score_of_doc)
    
    gram_vocab_df['Precision_0'] = precision_0
    gram_vocab_df['Recall_0'] = recall_0
    gram_vocab_df['F1_score_0'] = f1_score_0
    
    return gram_vocab_df      
        
def forExcel(data_df):
    data_df['No_of_Docs'] = data_df['Doc_Length']
    data_df.drop(['Doc_Length','Doc_Set'],axis=1,inplace=True)
    return data_df

##Amendment Reason
data=pd.read_csv("Lemmatized_Amend_Reason.csv")
cols=data.columns.tolist()
data=data[data.notnull()]

reason=data['Corrected_Reason_1_lemmatized_String']
reason_df_with_null = pd.DataFrame(list(reason), columns=['Reason_1'])
reason_df_with_null['Doc_ID'] = list(data.index)
reason_df_with_null['Renewed'] = data['Renewed']
reason_df = reason_df_with_null[reason_df_with_null['Reason_1'].notnull()]

gram_2_info = getNGramsInfo(reason_df,'Reason_1','Renewed','Doc_ID',2,50)
gram_3_info = getNGramsInfo(reason_df,'Reason_1','Renewed','Doc_ID',3,50)
gram_4_info = getNGramsInfo(reason_df,'Reason_1','Renewed','Doc_ID',4,50)
gram_5_info = getNGramsInfo(reason_df,'Reason_1','Renewed','Doc_ID',5,50)
gram_6_info = getNGramsInfo(reason_df,'Reason_1','Renewed','Doc_ID',6,50)
grams_6 = getAllGrams(reason_df,'Reason_1',6)
grams_7 = getAllGrams(reason_df,'Reason_1',7)

gram_2_info.to_csv("Amend_Reason_gram_2_info.csv",index=False)
gram_3_info.to_csv("Amend_Reason_gram_3_info.csv",index=False)
gram_4_info.to_csv("Amend_Reason_gram_4_info.csv",index=False)
gram_5_info.to_csv("Amend_Reason_gram_5_info.csv",index=False)
gram_6_info.to_csv("Amend_Reason_gram_6_info.csv",index=False)
grams_6.to_csv("Amend_Reason_gram_6.csv",index=False)
grams_7.to_csv("Amend_Reason_gram_7.csv",index=False)

#to do of files break while exporting to csv
ar_2 = forExcel(gram_2_info)
ar_3 = forExcel(gram_3_info)
ar_4 = forExcel(gram_4_info)
ar_5 = forExcel(gram_5_info)

ar_2.to_csv("Amend_Reason_gram_2_info.csv",index=False)
ar_3.to_csv("Amend_Reason_gram_3_info.csv",index=False)
ar_4.to_csv("Amend_Reason_gram_4_info.csv",index=False)
ar_5.to_csv("Amend_Reason_gram_5_info.csv",index=False)
grams_6.to_csv("Amend_Reason_gram_6.csv",index=False)
grams_7.to_csv("Amend_Reason_gram_7.csv",index=False)

##Transaction Notes
#change filepath accordingly
transact_data = pd.read_csv('Transaction Notes\\Stemmed_Transaction_Notes.csv')

t_notes=transact_data['Corrected_Notes_lemmatized']
transact_notes_with_null = pd.DataFrame(list(t_notes), columns=['Notes'])
transact_notes_with_null['Doc_ID'] = list(transact_data.index)
transact_notes_with_null['Renewed'] = transact_data['Renewed']
transact_notes_df = transact_notes_with_null[transact_notes_with_null['Notes'].notnull()]


notes_gram_2_info = getNGramsInfo(transact_notes_df,'Notes','Renewed','Doc_ID',2,50)
notes_gram_3_info = getNGramsInfo(transact_notes_df,'Notes','Renewed','Doc_ID',3,50)
notes_gram_4_info = getNGramsInfo(transact_notes_df,'Notes','Renewed','Doc_ID',4,50)
notes_gram_5_info = getNGramsInfo(transact_notes_df,'Notes','Renewed','Doc_ID',5,50)
notes_gram_6_info = getNGramsInfo(transact_notes_df,'Notes','Renewed','Doc_ID',6,50)
notes_gram_7 = getAllGrams(transact_notes_df,'Notes',7)
notes_gram_8 = getAllGrams(transact_notes_df,'Notes',8)

#done as the files were breaking when exported to csv
tn_2 = forExcel(notes_gram_2_info)
tn_3 = forExcel(notes_gram_3_info)
tn_4 = forExcel(notes_gram_4_info)
tn_5 = forExcel(notes_gram_5_info)
tn_6 = forExcel(notes_gram_6_info)


tn_2.to_csv("Transaction Notes\\Transaction_Notes_gram_2_info.csv",index=False)
tn_3.to_csv("Transaction Notes\\Transaction_Notes_gram_3_info.csv",index=False)
tn_4.to_csv("Transaction Notes\\Transaction_Notes_gram_4_info.csv",index=False)
tn_5.to_csv("Transaction Notes\\Transaction_Notes_gram_5_info.csv",index=False)
tn_6.to_csv("Transaction Notes\\Transaction_Notes_gram_6_info.csv",index=False)
notes_gram_7.to_csv("Transaction Notes\\Transaction_Notes_gram_7.csv",index=False)
notes_gram_8.to_csv("Transaction Notes\\Transaction_Notes_gram_8.csv",index=False)
