
# Stemming and Lemmatization LossDesc
import pandas as pd
import string
import nltk
import re

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')

notes=dt1['LossDesc_Cleansed']

def stop_word_removal(text):
    email1 = text.lower()    
    # clean and tokenize document string
    email_content = email1.split()    
    word_list = []
    for i in email_content:
#        x = 0
        if (('http' not in i) and ('@' not in i) and ('<.*?>' not in i) and i.isalnum() and (not i in stopwords)):
            word_list += [i]
    filtered_tokens=[]            
    for token in word_list:
        if re.search('^[a-z]+$', token):
            filtered_tokens.append(token)            
    filtered_tokens = [i for i in filtered_tokens if i not in string.punctuation] # Extra Code not done by karan

    return filtered_tokens 

def tokenize_and_lematize(text):
    tokens = [word.lower () for word in nltk.word_tokenize(text)]
    
    filtered_tokens = []
    
    for token in tokens:
        if re.search('^[a-z]+$', token):
            filtered_tokens.append(token)
    return filtered_tokens            

removed_stop_words=[]
for note in notes:
       x1=stop_word_removal(note)
       removed_stop_words.append(x1)
#       
lemmatized_corpus=removed_stop_words

test2=[]            
for w in lemmatized_corpus:
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
    

from nltk.stem import LancasterStemmer
from nltk import stem
lancaster = stem.lancaster.LancasterStemmer()
#tokens =  ['player', 'playa', 'playas', 'pleyaz'] 
#li = [lancaster.stem(i) for i in tokens]
#
stemmer = LancasterStemmer("english")

def stem_lancaster(text):
    tokens = [word.lower () for word in nltk.word_tokenize(text)]
    filtered_tokens = []
    
    for token in tokens:
        if re.search('^[a-z]+$', token):
            filtered_tokens.append(token)
#    stems=[]        
#    for t in filtered_tokens:
#        t_stem=lancaster.stem(t)             
#        stems.append(t_stem)    
    stems = [lancaster.stem(t) for t in filtered_tokens]
    return stems

lemmatized_corpus=test2
dt1['Corrected_LossDesc_lemmatized_New']=lemmatized_corpus
dt1['Corrected_LossDesc_lemmatized_String_New'] = list(map(' '.join, dt1['Corrected_LossDesc_lemmatized_New']))
stem = dt1['Corrected_LossDesc_lemmatized_String_New']
stem=stem[stem.notnull()]
stemmed_corpus = []
for m in stem: 
    stem_v = stem_lancaster(m)
    stemmed_corpus.append(stem_v)
    
dt1['Corrected_LossDesc_Stemmed_New']=stemmed_corpus

dt1['Corrected_LossDesc_Stemmed_String_New'] = list(map(' '.join, dt1['Corrected_LossDesc_Stemmed_New']))
dt1.columns
dt1 = dt1.drop(['LossDesc', 'LossDesc_Cleansed',
       'Corrected_LossDesc_lemmatized_New',
       'Corrected_LossDesc_lemmatized_String_New',
       'Corrected_LossDesc_Stemmed_New'],1)
dt1['LossDesc_Cleansed_Final']    = dt1['Corrected_LossDesc_Stemmed_String_New']
dt1 = dt1.drop(['Corrected_LossDesc_Stemmed_String_New'],1)
    

#r = data['Corrected_LossDesc_lemmatized_String_New'].value_counts()
#new_data.to_csv("D:\\Sentiment Use Case\\Transaction Notes\\Stemmed_Transaction_Notes4.csv",index=False)
new_data = pd.merge(data_reason,dt1,how='left',on='Policy')
new_data.to_csv("D:\\Sentiment Use Case\\Test_data\\Stemmed_Lemmatized_Test_LossDesc.csv",index=False)
