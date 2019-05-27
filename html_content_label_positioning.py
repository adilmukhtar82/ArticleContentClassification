import pandas as pd
from bs4 import BeautifulSoup
import requests
from bs4.element import Comment
import urllib.request
import re
import string
from nltk.tokenize import RegexpTokenizer

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

df = pd.read_excel('cincinnati for keyword postioning model eval.xlsx', header=None)
#print(df[0].values)

keyword_position_dict = dict()
keyword_position_dict['position_keyword'] = list()
keyword_position_dict['sponsored'] = list()
urls_id = 1
for url, label in zip(df[0].values, df[1].values):
    #if label == 'Yes':
        # r = requests.get(url)
        # soup = BeautifulSoup(r.content)
        # print(soup)
        # input()
    try:
        print(url)
        html = urllib.request.urlopen(url, timeout=5).read()
        raw_content = re.sub(' +', ' ', text_from_html(html).strip().lower())
        raw_content = ' '.join(tokenizer.tokenize(raw_content))

        # for kw in model_key_words:
        #     print(kw, raw_content.find(kw), len(raw_content))
        keyword_position = 0
        for kw in model_key_words:
            if kw in raw_content:

                keyword_position = (raw_content.find(kw))/len(raw_content)
                print('Model keywords found...!', kw, keyword_position, label)
                break

        keyword_position_dict['position_keyword'].append(keyword_position)

        if label == 'No':
            keyword_position_dict['sponsored'].append(0)
        else:
            keyword_position_dict['sponsored'].append(1)
                #print(kw, raw_content.split(' ').index(kw.split(' ')[0]), len(raw_content.split(' ')), raw_content.split(' ').index(kw.split(' ')[0])/len(raw_content.split(' ')))
        with open('content file cincinnati\\'+label+'-'+str(urls_id)+'.txt', 'w') as f:
            f.write(raw_content)
        print('Links completed:', urls_id)
        urls_id += 1
    except:
        continue

kw_positioning_df = pd.DataFrame(keyword_position_dict)
kw_positioning_df.to_csv('keyword_position_dataset_CINCINNATI.csv', sep=',', encoding='utf-8')