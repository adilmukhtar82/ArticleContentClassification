import os
import nltk
import pandas as pd
sponsored_dir = 'Preprocessed sponsored links'


pos_taggers_dict = {}
pos_taggers_dict['verbs'] = list()
pos_taggers_dict['nouns'] = list()
pos_taggers_dict['adjectives'] = list()
pos_taggers_dict['words_count'] = list()
pos_taggers_dict['adverbs'] = list()
pos_taggers_dict['preposition'] = list()


verbs = ['VB', 'VBD', 'VBP', 'VBZ', 'VBG', 'VBN']
nouns = ['NN', 'NNS', 'NNP', 'NNPS']
adjectives = ['JJ', 'JJS', 'JJR']
adverbs = ['RB', 'RBS', 'RBR']
preposition = ['IN']

for file in os.listdir(sponsored_dir):
    print(file)
    with open(sponsored_dir+'\\'+file, 'r') as f:
        #print(f.readlines()[0])
        try:
            pos_tagged = nltk.pos_tag(f.readlines()[0].split(' '))
            v = prep = adj = adv = n = 0
            for p in pos_tagged:
                if p[1] in verbs:
                    v += 1
                if p[1] in nouns:
                    n += 1
                if p[1] in adjectives:
                    adj += 1
                if p[1] in adverbs:
                    adv += 1
                if p[1] in preposition:
                    prep += 1
            pos_taggers_dict['verbs'].append(v)
            pos_taggers_dict['nouns'].append(n)
            pos_taggers_dict['adjectives'].append(adj)
            pos_taggers_dict['adverbs'].append(adv)
            pos_taggers_dict['preposition'].append(prep)
            pos_taggers_dict['words_count'].append(len(pos_tagged))
            print(pos_taggers_dict)
        except:
            continue
df = pd.DataFrame(pos_taggers_dict)

df.to_csv('bag of POS.csv',sep=',', encoding='utf-8')