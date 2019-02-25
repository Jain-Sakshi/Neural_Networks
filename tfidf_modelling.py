# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 18:55:52 2018

@author: sakshij
"""

"""
tfidf for Amend Reason
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

from sklearn.feature_extraction.text import TfidfVectorizer

def combineTokens(a):
    ngrams = []
    for gram_data_frame in a:
#        print(gram_data_frame['NGram'].values)
        raw_value = gram_data_frame['NGram'].values
        token_values = [' '.join(token) for token in raw_value]
        ngrams.append(token_values)
#        print('1')
    return flatten(ngrams)
        

amend_reason = pd.read_csv("Amendment Reason\\Lemmatized_Amend_Reason.csv")
transaction_notes = pd.read_csv('Transaction Notes\\Stemmed_Transaction_Notes.csv')

##########################################################################
##tfidf
tfidf_vect_amend_reason = TfidfVectorizer(analyzer='word', ngram_range=(2,5))
amend_corpus = amend_reason['Corrected_Reason_1_lemmatized_String'][amend_reason['Corrected_Reason_1_lemmatized_String'].notnull()]
tfidf_vect_amend_reason.fit(amend_corpus)
amend_reason_mapping = dict(tfidf_vect_amend_reason.vocabulary_)
stop_words_set_amend_reason = list(tfidf_vect_amend_reason.stop_words_)

tfidf_vect_transaction_notes = TfidfVectorizer(analyzer='word', ngram_range=(2,6))
transact_corpus = transaction_notes['Corrected_Notes_lemmatized'][transaction_notes['Corrected_Notes_lemmatized'].notnull()]
tfidf_vect_transaction_notes.fit(transact_corpus)
transaction_notes_mapping = dict(tfidf_vect_transaction_notes.vocabulary_)

stop_words_set_transaction_notes = list(tfidf_vect_transaction_notes.stop_words_)

combined_tokens_amend = combineTokens([gram_2_info, gram_3_info, gram_4_info, gram_5_info])
combined_tokens_transaction = combineTokens([notes_gram_2_info, notes_gram_3_info, notes_gram_4_info, notes_gram_5_info, notes_gram_6_info])

amend_final_tokens_mapping = dict([(k, amend_reason_mapping.get(k)) for k in combined_tokens_amend])
transaction_final_tokens_mapping = dict([(k, transaction_notes_mapping.get(k)) for k in combined_tokens_transaction])