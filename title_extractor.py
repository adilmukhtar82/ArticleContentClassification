# -*- coding: utf-8 -*-
from goose import Goose
import pandas as pd

from pandas import ExcelWriter
from pandas import ExcelFile

from time import sleep

Write_Excel_Filename = 'features_dataset.xlsx'
Read_Excel_Filename = 'DataSet.xlsx'
Sheet_Name = 'CIO'

Currency_Signs = {'$','€','£'}

Links_List = []
Title_List = []
Meta_Description_List = []
Body_List = []
Labels_List = []
Avg_Sents_Len_List = []
Avg_Words_Len_List = []
Curr_Signs_Count_List = []
Href_Freq_List = []

df = pd.read_excel(Read_Excel_Filename, sheetname=Sheet_Name)
for i in df.index:
    print(i)
    Labels_List.append(df['Labels'][i])
    url = df['Links'][i]

    Links_List.append(url)

    g = Goose()
    article = g.extract(url=url)
    #print (dir(article))

    Title_List.append(article.title)
    Meta_Description_List.append(article.meta_description)
    Body = article.cleaned_text
    Body_List.append(Body)

    sents = Body.split('.')
    words = Body.split()
    avg_sents_len = sum(len(x.split()) for x in sents) / len(sents)
    Avg_Sents_Len_List.append(avg_sents_len)
    avg_words_len = sum(len(word) for word in words) / len(words)
    Avg_Words_Len_List.append(avg_words_len)
    curr_signs_count = sum(Body.count(sign.decode('utf-8')) for sign in Currency_Signs)
    Curr_Signs_Count_List.append(curr_signs_count)
    href_freq = len(article.links)
    Href_Freq_List.append(href_freq)

    if i == 2:
    	break


'''
print(len(Links_List))
print(len(Title_List))
print(len(Meta_Description_List))
print(len(Body_List))
print(len(Labels_List))
print(len(Avg_Sents_Len_List))
print(len(Avg_Words_Len_List))
print(len(Curr_Signs_Count_List))
print(len(Href_Freq_List))
'''

StatusFrame = pd.DataFrame({'Links':Links_List, 'Title':Title_List , 'Meta': Meta_Description_List, 'Body':Body_List, 'Average Sentence Length':Avg_Sents_Len_List,
	'Average Words Length':Avg_Words_Len_List,'Currency Signs Count':Curr_Signs_Count_List, 'href Frequency': Href_Freq_List, 'Labels':Labels_List})
writer = ExcelWriter(Write_Excel_Filename,engine='openpyxl')
StatusFrame.to_excel(writer,Sheet_Name,index=False)
writer.save()
