
import os
import nltk
import pandas as pd
from nltk import word_tokenize
import os
import pandas as pd
import urllib
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib import request
import nltk
import sumy
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from nltk import SnowballStemmer
from nltk import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
def clean(cont):
    html_tag = '<[^<>]+>'
    number_tag = '[0-9]+'
    url_tag = '(http|https)://[^\s]*'
    url_head = '(http|https)'
    emailaddr_tag = '[^\s]+@[^\s]+'
    currency_tag = '[$]+'
    alphanumeric_tag = '[^a-zA-Z0-9]'

    cont = cont.lower()
    '''
    % Strip all HTML
    % Looks for any expression that starts with < and ends with > and replace
    % and does not have any < or > in the tag it with a space
    '''
    cont = re.sub(html_tag, ' ', cont)

    '''
    % Handle Numbers
    % Look for one or more characters between 0-9
    '''
    cont = re.sub(number_tag, ' ', cont)

    '''
    % Handle URLS
    % Look for strings starting with http:// or https://
    '''
    cont = re.sub(url_tag, ' ', cont)

    cont = re.sub(url_head, ' ', cont)

    '''
    % Handle Email Addresses
    % Look for strings with @ in the middle
    '''
    cont = re.sub(emailaddr_tag, ' ', cont)
    return cont
link_id = 1

pos_taggers_dict = {}
pos_taggers_dict['vb'] = list()
pos_taggers_dict['vbd'] = list()
pos_taggers_dict['vbp'] = list()
pos_taggers_dict['vbz'] = list()
pos_taggers_dict['vbg'] = list()
pos_taggers_dict['vbn'] = list()
pos_taggers_dict['nn'] = list()
pos_taggers_dict['nns'] = list()
pos_taggers_dict['nnp'] = list()
pos_taggers_dict['nnps'] = list()
pos_taggers_dict['jj'] = list()
pos_taggers_dict['jjs'] = list()
pos_taggers_dict['jjr'] = list()
pos_taggers_dict['rb'] = list()
pos_taggers_dict['rbs'] = list()
pos_taggers_dict['rbr'] = list()
pos_taggers_dict['in'] = list()
pos_taggers_dict['sponsored'] = list()

LANGUAGE = "english"
files_dir = 'NLP Preprocessed links Computer world\\'
for file in os.listdir(files_dir):
    try:
        with open(files_dir+file, 'r') as f:
            content = f.read()
        content = clean(content)
        # list_of_words = remove_stopwords(list_of_words, lang="english")
        # list_of_words = stemming(list_of_words, type="WordNetLemmatizer")
        # content = b' '.join(list_of_words).decode()
        #print(content)
        #print(file.split('-'))
        pos_tagged = nltk.pos_tag(word_tokenize(content))
        #print(pos_tagged)
        vb = vbd = vbp = vbz = vbg = vbn = nn = nnp = nns = nnps = jj = jjs = jjr = rb = rbs = rbr = in_prep = 0
        for p in pos_tagged:
            if p[1].lower() == 'vb':
                vb += 1
            if p[1].lower() == 'vbd':
                vbd += 1
            if p[1].lower() == 'vbp':
                vbp += 1
            if p[1].lower() == 'vbz':
                vbz += 1
            if p[1].lower() == 'vbg':
                vbg += 1
            if p[1].lower() == 'vbn':
                vbn += 1
            if p[1].lower() == 'nn':
                nn += 1
            if p[1].lower() == 'nnp':
                nnp += 1
            if p[1].lower() == 'nns':
                nns += 1
            if p[1].lower() == 'nnps':
                nnps += 1
            if p[1].lower() == 'jj':
                jj += 1
            if p[1].lower() == 'jjs':
                jjs += 1
            if p[1].lower() == 'jjr':
                jjr += 1
            if p[1].lower() == 'rb':
                rb += 1
            if p[1].lower() == 'rbs':
                rbs += 1
            if p[1].lower() == 'rbr':
                rbr += 1
            if p[1].lower() == 'in':
                in_prep += 1
        pos_taggers_dict['vb'].append(vb)
        pos_taggers_dict['vbd'].append(vbd)
        pos_taggers_dict['vbp'].append(vbp)
        pos_taggers_dict['vbz'].append(vbz)
        pos_taggers_dict['vbg'].append(vbg)
        pos_taggers_dict['vbn'].append(vbn)
        pos_taggers_dict['nn'].append(nn)
        pos_taggers_dict['nns'].append(nns)
        pos_taggers_dict['nnp'].append(nnp)
        pos_taggers_dict['nnps'].append(nnps)
        pos_taggers_dict['jj'].append(jj)
        pos_taggers_dict['jjs'].append(jjs)
        pos_taggers_dict['jjr'].append(jjr)
        pos_taggers_dict['rb'].append(rb)
        pos_taggers_dict['rbs'].append(rbs)
        pos_taggers_dict['rbr'].append(rbr)
        pos_taggers_dict['in'].append(in_prep)
        if file.split('-')[0] == '0':
            pos_taggers_dict['sponsored'].append(0)
        else:
            pos_taggers_dict['sponsored'].append(1)
        #print(pos_taggers_dict)
        print('Links completed:', link_id)
        link_id += 1
    except Exception as e:
        print('Exception:', e)

df = pd.DataFrame(pos_taggers_dict)

df.to_csv('bag of POS CW.csv', sep=',', encoding='utf-8')