'''
Description:    This file can be used for text pre-processing to feature vector conversion in NLP 
'''


import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from collections import Counter
import glob
import os
import operator
import numpy as np

# Global varibale that will be used by build_dict() to maintain the words dictionary
WordsDictionary = dict()

# This function will take the clean email (list of words) and add each word to dictionary
def build_dict(email):
    for word in email:
        if word not in WordsDictionary:
            WordsDictionary[word] = 1
        else:
            WordsDictionary[word] += 1

# This function will take a raw email, clean it and use build_dict() to add unique words 
def clean(email):
    html_tag = '<[^<>]+>'
    number_tag = '[0-9]+'
    url_tag = '(http|https)://[^\s]*'
    emailaddr_tag ='[^\s]+@[^\s]+'
    currency_tag = '[$]+'
    alphanumeric_tag = '[^a-zA-Z0-9]'

    
    email = email.lower()
    '''
    % Strip all HTML
    % Looks for any expression that starts with < and ends with > and replace
    % and does not have any < or > in the tag it with a space
    '''
    email = re.sub(html_tag,' ', email)

    '''
    % Handle Numbers
    % Look for one or more characters between 0-9
    '''
    email = re.sub(number_tag,'number', email)

    '''
    % Handle URLS
    % Look for strings starting with http:// or https://
    '''
    email = re.sub(url_tag,'httpaddr', email)

    '''
    % Handle Email Addresses
    % Look for strings with @ in the middle
    '''
    email = re.sub(emailaddr_tag,'emailaddr', email)

    '''
    Handle $ sign
    '''
    email = re.sub(currency_tag,'dollar', email)

    '''
    % Remove any non alphanumeric characters
    '''
    #email = re.sub(alphanumeric_tag,'', email) 

    tokenizer = RegexpTokenizer(r'\w+')
    email = tokenizer.tokenize(email)
     
    ps = PorterStemmer()
    #email = [ps.stem(word).encode('utf-8') for word in email]
    email = [ps.stem(word) for word in email]
    build_dict(email)
    email = ' '.join(email)
    return email    

# This function will clean complete training directory and and write
# all clean emails into 'clean/' directory
def clean_training_data(train_dir):
    for f in glob.glob(train_dir + "*.txt"):
        fd = open(f,'r')
        email = fd.read()
        email = clean(email)
        fd.close()

        clean_dir = train_dir + 'clean/'
        if not os.path.exists(clean_dir):
            os.makedirs(clean_dir)
        fd = open(clean_dir + f.split('\\')[-1:][0],"w")
        fd.write(str(email))
        fd.close()

def writeVocabList(dictionary):
    fd = open('vocab.txt','w')
    VocabList = dictionary.keys()
    for word in VocabList:
        fd.write(str(word)+'\n')
    fd.close()

def readVocabList(filename):
    with open(filename) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
    VocabList = [x.strip() for x in content] 
    f.close()
    return VocabList

def word_to_vocab_indices(email,VocabList):
    word_indices = []
    for word in email:
        if word in VocabList:
            #print VocabList.index(word)
            word_indices.append(VocabList.index(word))
        else:
            "Not in Vocab list"
            continue
    return word_indices

def get_feature_vector(word_indices,n_dict_words):

    features = np.zeros(shape=(n_dict_words,1))
    features[word_indices] = 1

    return features
    
if __name__ == "__main__":
    path_to_read_data = 'data/train/'
    clean_training_data(path_to_read_data)

    
    counts = WordsDictionary.values()
    keys = WordsDictionary.keys()

    #print (sum(list(counts)))
    print (sum(list(counts)), ' total words ', len(keys), ' unique words')

    
    # Select Most Common Words for training, choose top 1000 frequent words or more
    WordsDictionary = Counter(WordsDictionary)
    WordsDictionary = WordsDictionary.most_common(1000)

    writeVocabList(dict(WordsDictionary))
    VocabList = readVocabList('vocab.txt')


    
    fd = open('data/train/clean/sponsored_1.txt','r')
    email = fd.read()
    fd.close()
    email = email.split(' ')
    #print (email)

    word_indices = word_to_vocab_indices(email,VocabList)
    print (word_indices)
    features = get_feature_vector(word_indices,len(VocabList))
    print (features.shape)
    
