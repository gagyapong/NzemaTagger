import os
import re
from bs4 import BeautifulSoup

# CONSTANTS

taglist = ['PRON', 'SCONJ', 'ADV', 'DET', 'INTJ', 'X', 'PART', 'ADP',
                'SYM', 'NUM', 'CCONJ', 'VERB', 'PROPN', 'ADJ', 'NOUN']  

# TRAINING 

def preprocess_train_data():
    '''
    Returns a ConditionalFreqDist with all instances of previously annotated data
    Untagged words present in the XML file (mostly punctuation words) are ommitted
    '''

    current_dir = os.getcwd()
    data_file_path = os.path.join(current_dir, 'data/train/annotated-words-nzi.xml')

    return parse_training_xml(data_file_path)

def parse_training_xml(filename):
    '''
    Takes in an .xml filename path
    Creates and returns a list of annotated sentences in the form [(word, tag), ...]
    '''

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
    
# TEST

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

# GOLDEN STANDARDS

def preprocess_gs_data():
    '''
    Returns a dictionary of preprocessed sentences
    The {key : value} pairs are as follows -- {filename : list of sentences}
    '''

    current_dir = os.getcwd()
    data_file_path = os.path.join(current_dir, 'data/gs/gs-nzi.xml')

    return parse_gs_xml(data_file_path)

def parse_gs_xml(filename):
    '''
    Takes in an .xml filename path
    Returns a dictionary of preprocessed sentences
    The {key : value} pairs are as follows -- {filename : list of sentences}
    '''
    
    with open(filename, 'r') as file:
        data = file.read()

        bs_data = BeautifulSoup(data, 'xml')
        tagged_sentences = []

        entries = bs_data.find_all('w:rStyle', {'w:val': 'Interlin Base nzi'})
        raw_tags = []
        
        # Code snippet adapted from ChatGPT
        for tag in entries:
            base_text_tag = tag.find_next('m:t')
            pos_tag = tag.find_next('w:rStyle', {'w:val': 'Interlin Word POS'})
            pos_text_tag = pos_tag.find_next('m:t') if pos_tag is not None else None

            # Removing header
            if base_text_tag.text.strip() == "PASSAGE":
                continue

            # Removing whitespace at the end
            raw_tags.append((base_text_tag.text.strip(), pos_text_tag.text.strip() if pos_text_tag is not None else None))

        gs_tag_dict = {"passage1.txt": [], "passage2.txt": []}

        # Format tuples into sentences
        sent = []
        for tup in raw_tags[1:]:
            # Guard clause to check for passage 2
            if re.search('2', tup[0]):
                gs_tag_dict['passage1.txt'] = tagged_sentences
                tagged_sentences = []
                continue

            if not re.search(r'[^\w\s]', tup[0]):
                    sent.append(tup)

            elif tup[0].endswith('.'):
                tagged_sentences.append(sent)
                sent = []
                continue
            else:
                continue

        gs_tag_dict['passage2.txt'] = tagged_sentences

        return gs_tag_dict