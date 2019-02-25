# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:33:32 2018

@author: sakshij
"""

import pandas as pd
import nltk
from nltk.metrics import precision, recall, f_measure
from nltk.collocations import *
from itertools import chain
from nltk.util import bigrams
import numpy as np
import re
from nltk.corpus import stopwords 
from collections import Counter
from nltk.stem.porter import *
import ast
from nltk.corpus import wordnet as wn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import os, sys, email
import gensim
from gensim.models import Doc2Vec
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
from string import punctuation
import timeit
from sklearn.cluster import KMeans
from sklearn import metrics
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from nltk.stem import WordNetLemmatizer


def flatten(two_dim_list):
    return list(chain.from_iterable(two_dim_list))

def dropPolicies(data, data_coln):
#    initial = data_df[data_coln]
#    only_alphabets = initial.apply(lambda x : re.sub('[^a-zA-Z]+', ' ', x))
#    list_of_words = only_alphabets.apply(lambda x : x.split(' '))
#    list_of_proper_words = list_of_words.apply(lambda x : [word for word in x if len(word) > 2 and word not in stopwords])
#    list_of_proper_words_joined = list_of_proper_words.apply(lambda x : ' '.join(x))
    
    data[data_coln] = data[data_coln].apply(lambda x : str(x).strip() )
    data[data_coln] = data[data_coln].apply(lambda x : re.sub('[^a-zA-Z \n\.]', '', str(x)))
    data.replace(r'^\s*$', np.NaN, regex=True, inplace = True)
    data['Corrected_Reason_2'] =  data[data_coln]
    data['Corrected_Reason_2'] = data['Corrected_Reason_2'].apply(lambda x : str(x).strip() )
    data['Corrected_Reason_2'] =  data['Corrected_Reason_2'].apply(lambda x : len(str(x).split(' ')))
    new_data= data[data['Corrected_Reason_2']!=1]
    new_data = new_data[new_data[data_coln].notnull()]
    
    return new_data
    
    
#    stopwords_list = list(set(stopwords.words('english')))
#    new_data[cleansed_coln] = new_data[data_coln].apply(lambda x : [word for word in x.split(' ') if len(word) > 2 and word not in stopwords_list])
#    stemmer = PorterStemmer()
#    new_data[cleansed_coln] = new_data[cleansed_coln].apply(lambda x : [stemmer.stem(token) for token in x])
#    new_data[cleansed_coln+ '_joined'] = new_data[cleansed_coln].apply(lambda x : ' '.join(x))

def stop_word_removal(text):
    tokens = [word.lower() for word in nltk.word_tokenize(text)]
    filtered_tokens = []
    lemmatizer = WordNetLemmatizer()
    
    for token in tokens:
        if re.search('^[a-z]+$', token):
            filtered_tokens.append(token)
    
    test2=[]            
    for w in filtered_tokens:
        test1=[]    
        for each in w:
            try:
                tmp = wn.synsets(each)[0].pos()
                print (each, ":", tmp)
            except:
                print(each)
                tmp = 'n'            
            stems =lemmatizer.lemmatize(each,pos=tmp)
            test1.append(stems)
        test2.append(test1)
    return stems

def getAllUnigramInfo(data_df, data_coln, target_coln):
#    split_data = data_df[data_coln].apply(lambda x : x.split(' '))
    vocab_temp = list(set(flatten(data_df[data_coln])))
    vocab = list(filter(lambda name: name.strip(), vocab_temp))
    vocab_df = pd.DataFrame()
    vocab_df['Token'] = vocab
    
    data_df_1 = data_df[data_coln][data_df[target_coln] == 1]
    
    count_matrix_columns = data_df.index
    count_matrix = pd.DataFrame(0, columns = count_matrix_columns, index = vocab)
    
    for doc_id in count_matrix_columns:
        unigram_counter = dict(Counter(data_df[data_coln][doc_id]))
        for word in unigram_counter:
            count_matrix[doc_id][word] = unigram_counter[word]
        
    total_1_docs = len(data_df_1)
    all_1_docs = set(data_df_1.index)
    precision = []
    recall = []
    f1_score = []
    for word in vocab:
        phrase_nonzero = set(count_matrix.loc[word].nonzero()[0])
        r1_with_phrase = phrase_nonzero.intersection(all_1_docs)
        precision_of_doc = len(r1_with_phrase)/len(phrase_nonzero)
        recall_of_doc = len(r1_with_phrase)/total_1_docs
        try:
            f1_score_of_doc = (2*precision_of_doc*recall_of_doc)/(2*(precision_of_doc+recall_of_doc))
        except:
            f1_score_of_doc = -1
        
        precision.append(precision_of_doc)
        recall.append(recall_of_doc)
        f1_score.append(f1_score_of_doc)
    
    vocab_df['Precision'] = precision
    vocab_df['Recall'] = recall
    vocab_df['F1_score'] = f1_score
    
    return vocab_df

#data_df = corrected_amend_reason_cleaned
#data_coln = 'Cleansed_Reason_1'
#target_coln = 'Renewed'

def getAllBigramInfo(data_df, data_coln, target_coln):
    
    all_bigrams = []
    for doc in data_df[data_coln]:
        doc_bigrams = list(bigrams(doc))
        all_bigrams.append(doc_bigrams)
        
    data_df['Bigrams'] = all_bigrams
    all_bigrams_flattened = flatten(all_bigrams)
    bigram_vocab = list(set(all_bigrams_flattened))
    vocab_df = pd.DataFrame()
    vocab_df['Bigrams'] = bigram_vocab
    
    data_df_1 = data_df[data_coln][data_df[target_coln] == 1]
    
    count_matrix_columns = data_df.index
    count_matrix = pd.DataFrame(0, columns = count_matrix_columns, index = bigram_vocab)
    
    data_df_columns = list(data_df.columns)
    del_columns = [x for x in data_df_columns if x != 'Bigrams']
    data_df.drop(del_columns, axis=1, inplace=True)
    
    for doc_id in count_matrix_columns:
        bigram_counter = dict(Counter(data_df['Bigrams'][doc_id]))
        for word in bigram_counter:
            count_matrix[doc_id][word] = bigram_counter[word]
        
    total_1_docs = len(data_df_1)
    all_1_docs = set(data_df_1.index)
    precision = []
    recall = []
    f1_score = []
    
    for word in bigram_vocab:
        t = count_matrix.loc[[word]].transpose()
        t1 = pd.Series(t[word])
        phrase_nonzero = set(t1.nonzero()[0])
        r1_with_phrase = phrase_nonzero.intersection(all_1_docs)
        precision_of_doc = len(r1_with_phrase)/len(phrase_nonzero)
        recall_of_doc = len(r1_with_phrase)/total_1_docs
        try:
            f1_score_of_doc = (2*precision_of_doc*recall_of_doc)/(2*(precision_of_doc+recall_of_doc))
        except:
            f1_score_of_doc = -1
        
        precision.append(precision_of_doc)
        recall.append(recall_of_doc)
        f1_score.append(f1_score_of_doc)
    
    vocab_df['Precision'] = precision
    vocab_df['Recall'] = recall
    vocab_df['F1_score'] = f1_score
    
    return vocab_df

def getAllBigramInfoMemory(data_df, data_coln, target_coln):
    
    all_bigrams = []
    for doc in data_df[data_coln]:
        doc_bigrams = list(bigrams(doc))
        all_bigrams.append(doc_bigrams)
        
    data_df['Bigrams'] = all_bigrams
    all_bigrams_flattened = flatten(all_bigrams)
    bigram_vocab = list(set(all_bigrams_flattened))
    vocab_df = pd.DataFrame()
    vocab_df['Bigrams'] = bigram_vocab
    
    data_df_1 = data_df[data_coln][data_df[target_coln] == 1]
    
    data_df_columns = list(data_df.columns)
    del_columns = [x for x in data_df_columns if x != 'Bigrams']
    data_df.drop(del_columns, axis=1, inplace=True)
    
    bigram_occuring_documents_list = []
    for bigram in bigram_vocab:
        one_bigram_list = []
        for doc in data_df.index:
            if bigram in data_df['Bigrams']:
                one_bigram_list.append(doc)
        bigram_occuring_documents_list.append(one_bigram_list)
        
    total_1_docs = len(data_df_1)
    all_1_docs = set(data_df_1.index)
    precision = []
    recall = []
    f1_score = []
    
    for word_index in range(len(bigram_vocab)):
        phrase_nonzero = set(bigram_occuring_documents_list[word_index])
        r1_with_phrase = phrase_nonzero.intersection(all_1_docs)
        precision_of_doc = len(r1_with_phrase)/len(phrase_nonzero)
        recall_of_doc = len(r1_with_phrase)/total_1_docs
        try:
            f1_score_of_doc = (2*precision_of_doc*recall_of_doc)/(2*(precision_of_doc+recall_of_doc))
        except:
            f1_score_of_doc = -1
        
        precision.append(precision_of_doc)
        recall.append(recall_of_doc)
        f1_score.append(f1_score_of_doc)
    
    vocab_df['Precision'] = precision
    vocab_df['Recall'] = recall
    vocab_df['F1_score'] = f1_score
    
    return vocab_df


#Reading Data
corrected_amend_reason = pd.read_csv('Amendment Reason\\Correct_Amend_Reason.csv')
corrected_transaction_notes = pd.read_csv('Transaction Notes\\Corrected_Transaction_Notes.csv')

#Cleansing
corrected_amend_reason_cleaned['Corrected_Reason_1']
corrected_amend_reason_cleaned = dropPolicies(corrected_amend_reason,'Corrected_Reason_1')
#corrected_amend_reason_cleaned['Cleansed_Reason_1'] = stop_word_removal(corrected_amend_reason['Corrected_Reason_1'])

temp = []
for document in corrected_amend_reason['Corrected_Reason_1']:
    temp.append(stop_word_removal(document))
corrected_amend_reason_cleaned['Cleansed_Reason_1'] = temp

corrected_transaction_notes_cleaned = dropPolicies(corrected_transaction_notes,'Corrected_Notes')
corrected_transaction_notes_cleaned['Cleansed_Notes'] = stop_word_removal(corrected_transaction_notes['Corrected_Notes'])


corrected_amend_reason_cleaned.to_csv('Unigrams_Bigrams\\Cleaned_Amend_Reason.csv',index=False)
corrected_transaction_notes_cleaned.to_csv('Unigrams_Bigrams\\Cleaned_Transaction_Notes.csv',index=False)

#Unigrams
amend_reason_uni_info = getAllUnigramInfo(corrected_amend_reason_cleaned,'Cleansed_Reason_1','Renewed')
transaction_notes_uni_info = getAllUnigramInfo(corrected_transaction_notes_cleaned,'Cleansed_Notes','Renewed')

amend_reason_uni_info.to_csv('Unigrams_Bigrams\\Unigrams_Info_Amend_Reason.csv',index=False)
transaction_notes_uni_info.to_csv('Unigrams_Bigrams\\Unigrams_Info_Transaction_Notes.csv',index=False)

#Bigrams
corrected_amend_reason_cleaned['Cleansed_Reason_1'].apply(lambda x : ' '.join(x))
amend_reason_bi_info = getAllBigramInfoMemory(corrected_amend_reason_cleaned,'Cleansed_Reason_1','Renewed')
transaction_notes_bi_info = getAllBigramInfo(corrected_transaction_notes_cleaned,'Cleansed_Notes','Renewed')

amend_reason_bi_info.to_csv('Unigrams_Bigrams\\Bigrams_Info_Amend_Reason.csv',index=False)
transaction_notes_bi_info.to_csv('Unigrams_Bigrams\\Bigrams_Info_Transaction_Notes.csv',index=False)