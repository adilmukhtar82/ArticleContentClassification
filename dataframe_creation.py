# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from bs4 import BeautifulSoup
from sumy.parsers.html import HtmlParser
from bs4.element import Comment
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import pandas as pd
import re
import urllib.request
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import readability
import community


tokenizer = RegexpTokenizer(r'\w+')
model_key_words = ['brandpost', 'sponsored', 'brought to you by', 'story from', 'provided by']


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def text_from_html(body):
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)
    return u" ".join(t.strip() for t in visible_texts)

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

    cont = re.sub(' +', ' ', cont)
    return cont

def keyword_position(url):
    html = urllib.request.urlopen(url, timeout=5).read()
    raw_content = re.sub(' +', ' ', text_from_html(html).strip().lower())
    raw_content = ' '.join(tokenizer.tokenize(raw_content))

    # for kw in model_key_words:
    #     print(kw, raw_content.find(kw), len(raw_content))
    keyword_position = 0
    for kw in model_key_words:
        if kw in raw_content:
            keyword_position = (raw_content.find(kw)) / len(raw_content)
            break
    return keyword_position
LANGUAGE = "english"
SENTENCES_COUNT = 1000

links_file = pd.read_excel('CW CIO Links.xlsx', header=None)
count_id = 1

complete_dataframe = {}
complete_dataframe['vocab richness'] = list()
complete_dataframe['keyword position'] = list()
complete_dataframe['spaces'] = list()
complete_dataframe['tabs'] = list()
complete_dataframe['braces'] = list()
complete_dataframe['brackets'] = list()
complete_dataframe['words'] = list()
complete_dataframe['length text'] = list()
complete_dataframe['vb'] = list()
complete_dataframe['vbd'] = list()
complete_dataframe['vbp'] = list()
complete_dataframe['vbz'] = list()
complete_dataframe['vbg'] = list()
complete_dataframe['vbn'] = list()
complete_dataframe['nn'] = list()
complete_dataframe['nns'] = list()
complete_dataframe['nnp'] = list()
complete_dataframe['nnps'] = list()
complete_dataframe['jj'] = list()
complete_dataframe['jjs'] = list()
complete_dataframe['jjr'] = list()
complete_dataframe['rb'] = list()
complete_dataframe['rbs'] = list()
complete_dataframe['rbr'] = list()
complete_dataframe['in'] = list()
complete_dataframe['characters_per_word'] = list()
complete_dataframe['syll_per_word'] = list()
complete_dataframe['words_per_sentence'] = list()
complete_dataframe['sentences_per_paragraph'] = list()
complete_dataframe['type_token_ratio'] = list()
complete_dataframe['characters'] = list()
complete_dataframe['syllables'] = list()
complete_dataframe['wordtypes'] = list()
complete_dataframe['long_words'] = list()
complete_dataframe['complex_words'] = list()
complete_dataframe['complex_words_dc'] = list()
complete_dataframe['tobeverb'] = list()
complete_dataframe['auxverb'] = list()
complete_dataframe['conjunction'] = list()
complete_dataframe['pronoun'] = list()
complete_dataframe['preposition'] = list()
complete_dataframe['nominalization'] = list()
complete_dataframe['pronoun'] = list()
complete_dataframe['interrogative'] = list()
complete_dataframe['article'] = list()
complete_dataframe['subordination'] = list()
complete_dataframe['conjunction'] = list()
complete_dataframe['preposition'] = list()
complete_dataframe['Kincaid'] = list()
complete_dataframe['sponsored'] = list()


for url, label in zip(links_file[0], links_file[1]):
    try:
        parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
        stemmer = Stemmer(LANGUAGE)

        summarizer = Summarizer(stemmer)
        summarizer.stop_words = get_stop_words(LANGUAGE)
        print(url)
        keyword_pos = 0
        keyword_pos = keyword_position(url)



        with open('body text CW CIO\\' + str(label) + '-' + str(count_id) + '.txt', 'w') as f:
            content = ''
            sentences = summarizer(parser.document, SENTENCES_COUNT)
            # print('Total Sentence:', len(sentences))
            complete_para = ''

            for sent in sentences:
                complete_para += str(sent)+' '
                content = (content + str(sent)).lower()
            spaces_count = content.count(' ')
            tabs_count = content.count('\t')
            braces_count = content.count('{')
            brackets_count = content.count('[')
            words_count = len(re.split('\s+', content))
            length_text = len(content)
            content = clean(content)
            f.write(content)

        pos_tagged = nltk.pos_tag(word_tokenize(content))
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


        readability_dicts = readability.getmeasures(complete_para.split(' '))
        for dict in readability_dicts.values():
            for k, v in zip(dict.keys(), dict.values()):
                if k == 'Kincaid' or k == 'pronoun' or k == 'interrogative' or k == 'article' or k == 'subordination' or k == 'conjunction' or k =='preposition' or k == 'auxverb' or k == 'tobeverb' or k == 'conjunction' or k == 'pronoun' or k == 'preposition' or k == 'nominalization' or k == 'characters_per_word' or k == 'syll_per_word' or k == 'words_per_sentence' or k == 'sentences_per_paragraph' or k == 'type_token_ratio' or k == 'characters' or k == 'syllables' or k == 'wordtypes' or k == 'long_words' or k == 'complex_words' or k == 'complex_words_dc':
                    complete_dataframe[k].append(v)
        filtered_words = [word for word in content.split(' ') if word not in stopwords.words('english')]
        vocab_richness = len(set(list(filtered_words)))/len(list(filtered_words))
        complete_dataframe['vocab richness'].append(vocab_richness)
        complete_dataframe['keyword position'].append(keyword_pos)
        complete_dataframe['spaces'].append(spaces_count)
        complete_dataframe['tabs'].append(tabs_count)
        complete_dataframe['braces'].append(braces_count)
        complete_dataframe['brackets'].append(brackets_count)
        complete_dataframe['words'].append(words_count)
        complete_dataframe['length text'].append(length_text)
        complete_dataframe['vb'].append(vb)
        complete_dataframe['vbd'].append(vbd)
        complete_dataframe['vbp'].append(vbp)
        complete_dataframe['vbz'].append(vbz)
        complete_dataframe['vbg'].append(vbg)
        complete_dataframe['vbn'].append(vbn)
        complete_dataframe['nn'].append(nn)
        complete_dataframe['nns'].append(nns)
        complete_dataframe['nnp'].append(nnp)
        complete_dataframe['nnps'].append(nnps)
        complete_dataframe['jj'].append(jj)
        complete_dataframe['jjs'].append(jjs)
        complete_dataframe['jjr'].append(jjr)
        complete_dataframe['rb'].append(rb)
        complete_dataframe['rbs'].append(rbs)
        complete_dataframe['rbr'].append(rbr)
        complete_dataframe['in'].append(in_prep)
        if label == 'Yes':
            complete_dataframe['sponsored'].append(1)
        else:
            complete_dataframe['sponsored'].append(0)
        print('Url completed:', count_id)
        count_id += 1
    except:
        continue

df = pd.DataFrame(complete_dataframe)

df.to_csv('complete dataframe feature eng.csv', sep=',',
          encoding='utf-8')

