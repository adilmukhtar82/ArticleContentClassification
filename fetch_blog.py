# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import pandas as pd
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

    cont = re.sub(alphanumeric_tag, ' ', cont)

    cont = re.sub(currency_tag, ' ', cont)

    return cont

LANGUAGE = "english"
SENTENCES_COUNT = 1000

links_file = pd.read_excel('CW CIO Links.xlsx', header=None)
count_id = 1
for url, label in zip(links_file[0], links_file[1]):

    parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    print(url)
    with open('body text CW CIO\\'+str(label)+'-'+str(count_id)+'.txt', 'w') as f:
        content = ''
        sentences = summarizer(parser.document, SENTENCES_COUNT)
        #print('Total Sentence:', len(sentences))
        for sent in sentences:
            content = (content + str(sent)).lower()
        content = clean(content)
        f.write(content)
        print(content)


    print('Url completed:', count_id)
    count_id += 1
    input()
