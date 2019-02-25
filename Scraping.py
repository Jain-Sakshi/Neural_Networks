# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 17:00:15 2018

@author: harshal
"""

import pandas as pd
from bs4 import BeautifulSoup
#import requests
import urllib

############## First Page
req = urllib.request.Request(url ='https://www.elephant.com/car-insurance/terms',
                              headers={'User-Agent': 'Mozilla/5.0'})

html = urllib.request.urlopen(req).read()

soup = BeautifulSoup(html,'lxml')

headers =[]
event_containers = soup.find_all('div', class_ = "entry-content description clearfix")
#tab2 = soup.findAll(attrs={"class" : "entry-content description clearfix"})

sp = BeautifulSoup(event_containers,'lxml')

len(event_containers[0].find_all('h3'))
len(event_containers[0].find_all('h1'))
t3=event_containers[0].find_all('h3')
t1 =event_containers[0].find_all('h1')

h3_tag=[]
for tag in t3:
    t3=tag.text
    h3_tag.append(t3)
    
h1_tag=[]
for tag in t1:
    t1=tag.text
    h1_tag.append(t1)
    
dat = pd.DataFrame()
dat['Values']=h3_tag     
    
############### Second Page 
req1 = urllib.request.Request(url ='http://www.rda-insurance.com/insurance-agency-insurance-glossary.htm?sid=65124',
                              headers={'User-Agent': 'Mozilla/5.0'})

html1 = urllib.request.urlopen(req1).read()

soup1 = BeautifulSoup(html1,'lxml')

event_containers = soup1.find_all('div', class_ = "container")
    
headers = event_containers[1].find_all('strong')

header_tag=[]
for x in range(0,24):
    headers = event_containers[x].find_all('strong')
    for tag in headers:
        t3=tag.text
        header_tag.append(t3)
    
dat = dat.append(pd.DataFrame(header_tag, columns=['Values']))

######### Third Page
    
req2 = urllib.request.Request(url ='https://www.progressive.com/glossary/',
                              headers={'User-Agent': 'Mozilla/5.0'})

html2 = urllib.request.urlopen(req2).read()

soup2 = BeautifulSoup(html2,'lxml')
event_containers = soup2.find_all('dl', class_ = "group")
#headers = event_containers[14].find_all('dt')
header_tag_new=[]
for x in range(0,15):
#    print(x)
    headers = event_containers[x].find_all('dt')
    for tag in headers:
        t3=tag.text
        header_tag_new.append(t3)    

dat = dat.append(pd.DataFrame(header_tag_new, columns=['Values']))

############# Fourth Page

req3 = urllib.request.Request(url ='http://www.insurance.ca.gov/01-consumers/105-type/95-guides/01-auto/autoterms.cfm#u',
                              headers={'User-Agent': 'Mozilla/5.0'})

html3 = urllib.request.urlopen(req3).read()

soup3 = BeautifulSoup(html3,'lxml')
event_containers = soup3.find_all('div', class_ = "CS_Textblock_Text")

headers = event_containers[0].find_all('strong')

head_4=[]
for tag in headers:
    t3=tag.text
    head_4.append(t3)
head_4.remove('Back to Top')    
head_4.remove('')    
dat = dat.append(pd.DataFrame(head_4, columns=['Values']))

########## Fifth Page
req4 = urllib.request.Request(url ='https://www.einsurance.com/glossary/',
                              headers={'User-Agent': 'Mozilla/5.0'})

html4 = urllib.request.urlopen(req4).read()

soup4 = BeautifulSoup(html4,'lxml')
event_containers = soup4.find_all('div', class_ = "tab_inner_content invers-color")

#headers = event_containers[1].find_all('dt')
head_5=[]
for x in range(0,23):
    headers = event_containers[x].find_all('dt')
    for tag in headers:
        t3=tag.text
        head_5.append(t3)

dat = dat.append(pd.DataFrame(head_5, columns=['Values']))

############## Sixth Page

req5 = urllib.request.Request(url ='https://www.iselect.com.au/car/glossary/',
                              headers={'User-Agent': 'Mozilla/5.0'})

html5 = urllib.request.urlopen(req5).read()

soup5 = BeautifulSoup(html5,'lxml')
event_containers = soup5.find_all('div', class_ = "right-sidebar")

headers = event_containers[0].find_all('h3')
head_6=[]
#for x in range(0,23):
#    headers = event_containers[x].find_all('dt')
for tag in headers:
    t3=tag.text
    head_6.append(t3)
    
dat = dat.append(pd.DataFrame(head_6, columns=['Values']))

############## Seventh Page


req7 = urllib.request.Request(url ='https://www.esurance.ca/info/car-insights/glossary',
                              headers={'User-Agent': 'Mozilla/5.0'})

html7 = urllib.request.urlopen(req7).read()

soup7 = BeautifulSoup(html7,'lxml')
event_containers = soup7.find_all('div', class_ = "c9")

headers = event_containers[1].find_all('p', class_ = "title")
head_7=[]
#for x in range(0,23):
#    headers = event_containers[x].find_all('dt')
for tag in headers:
    t3=tag.text    
    head_7.append(t3)
    
dat = dat.append(pd.DataFrame(head_7, columns=['Values']))

################ Eigth Page

req8 = urllib.request.Request(url ='http://www.orrandassociates.ca/Tools-and-Resources/Glossary-Of-Terms',
                              headers={'User-Agent': 'Mozilla/5.0'})

html8 = urllib.request.urlopen(req8).read()

soup8 = BeautifulSoup(html8,'lxml')
event_containers = soup8.find_all('div', class_ = "container")
headers = event_containers[2].find_all('span', class_ = "GlossaryItem")

head_8=[]
for x in range(0,5):
    headers = event_containers[x].find_all('span', class_ = "GlossaryItem")
    for tag in headers:
        t3=tag.text    
        head_8.append(t3)

dat = dat.append(pd.DataFrame(head_8, columns=['Values']))

############# Ninth Page


req9 = urllib.request.Request(url ='https://www.geico.com/information/insurance-terms/',
                              headers={'User-Agent': 'Mozilla/5.0'})

html9 = urllib.request.urlopen(req9).read()

soup9 = BeautifulSoup(html9,'lxml')
event_containers = soup9.find_all('div', class_ = "col-lg-8")
headers = event_containers[0].find_all('p', class_ = "dictionary-term anchor h3")

head_9=[]
for x in range(0,1):
#    print(x)
    headers = event_containers[x].find_all('p', class_ = "dictionary-term anchor h3")
    for tag in headers:
        t3=tag.text    
        head_9.append(t3)

dat = dat.append(pd.DataFrame(head_9, columns=['Values']))

################ Tenth Page
req10 = urllib.request.Request(url ='https://www.carsdirect.com/car-insurance/glossary-of-car-insurance-terms-and-definitions',
                              headers={'User-Agent': 'Mozilla/5.0'})

html10 = urllib.request.urlopen(req10).read()

soup10 = BeautifulSoup(html10,'lxml')
event_containers = soup10.find_all('div', class_ = "entry-body default-layout")
headers = event_containers[0].find_all('strong')

head_10=[]
for x in range(0,1):
#    print(x)
    headers = event_containers[x].find_all('strong')
    for tag in headers:
        t3=tag.text    
        head_10.append(t3)

dat = dat.append(pd.DataFrame(head_10, columns=['Values']))

######################## Eleventh Page

req11 = urllib.request.Request(url ='https://www.amfam.com/resources/articles/understanding-insurance/car-insurance-terms-glossary',
                              headers={'User-Agent': 'Mozilla/5.0'})

html11 = urllib.request.urlopen(req11).read()

soup11 = BeautifulSoup(html11,'lxml')
event_containers = soup11.find_all('div', class_ = "rich-text ")
headers = event_containers[18].find_all('strong')

head_11=[]
for x in range(0,19):
#    print(x)
    headers = event_containers[x].find_all('strong')
    for tag in headers:
        t3=tag.text    
        head_11.append(t3)

mn = pd.DataFrame()

mn['val'] = head_11


mn['len'] = mn['val'].apply(lambda x : len(x))
mn1 = mn [mn['len']!=1]

head_11_1 = mn1['val']
dat = dat.append(pd.DataFrame(head_11_1, columns=['Values']))

############# Twelth Page
not_working
req12 = urllib.request.Request(url ='https://www.erieinsurance.com/auto-insurance/terms',
                              headers={'User-Agent': 'Mozilla/5.0'})

html12 = urllib.request.urlopen(req12).read()

soup12 = BeautifulSoup(html12,'lxml')
event_containers = soup12.find_all('div', class_ = "content-block container-fluid")

headers = event_containers[0].find_all('strong')




