import os
import re
from nltk import ConditionalFreqDist
from bs4 import BeautifulSoup

# HELPER FUNCTIONS

def tokenize(filename):
    '''
    Takes in a .txt filename path
    Returns a list of tokenized sentences (list of lists)
    '''
    sentences = []

    with open(filename, 'r') as file:
        sentences = file.readlines()

    sentences = [remove_punct(sent) for sent in sentences]  # Remove punctuation
    sentences = [sent.split() for sent in sentences]        # Tokenize

    return sentences

def remove_punct(sentence):
    return re.sub(r'[^\w\s]', '', sentence)

def read_xml_data(filename):
    '''
    Takes in an .xml filename path
    Creates and returns a list of annotated sentences in the form [(word, tag), ...]
    '''

    taglist = ['PRON', 'SCONJ', 'ADV', 'DET', 'INTJ', 'X', 'PART', 'ADP',
                    'SYM', 'NUM', 'CCONJ', 'VERB', 'PROPN', 'ADJ', 'NOUN']
    
    with open(filename, 'r') as file:
        data = file.read()

        bs_data = BeautifulSoup(data, 'xml')
        tagged_sentences = []

        for phrase in bs_data.find_all('phrase'):
            sent = []
            
            # Punctuation words are not annotated, thus we need to check for pairings
            words = phrase.words.text.split()
            for t1, t2 in zip (words[:-1], words[1:]):
                if t1 in taglist or t2 not in taglist:
                    continue
                else:
                    sent.append((t1, t2))
            
            tagged_sentences.append(sent)

        return tagged_sentences
                

# CONTENT FUNCTIONS 

def preprocess_train_data():
    '''
    Returns a ConditionalFreqDist with all instances of previously annotated data
    Untagged words present in the XML file (mostly punctuation words) are ommitted
    '''

    current_dir = os.getcwd()
    data_file_path = os.path.join(current_dir, 'data/train/annotated-words-nzi.xml')

    return read_xml_data(data_file_path)


def preprocess_test_data():
    '''
    Returns a dictionary of preprocessed sentences
    The {key : value} pairs are as follows -- {filename : list of sentences}
    '''

    # Get data folder path from current working directory
    current_dir = os.getcwd()
    data_file_path = os.path.join(current_dir, 'data/test')

    # Get list of files
    file_list = os.listdir(data_file_path)
    file_data = {}

    # Tokenize all files
    for file in file_list:
        path = os.path.join(data_file_path, file)
        file_data[file] = (tokenize(path))

    return file_data