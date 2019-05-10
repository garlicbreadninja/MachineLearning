import tensorflow as tf 
from sklearn.model_selection import train_test_split
import io 
import os
import numpy as np 
import time
import unicodedata
import matplotlib.pyplot as plt 
import re

path_to_zip = tf.keras.utils.get_file('spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    w = re.sub(r'([?.,!()¿])', r' \1 ', w)
    w = re.sub('/s+', r' ', w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w = '<start>' + w + '<end>'

    return w

def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    wordpairs = [[preprocess_sentence(w) for  w in l.split('\t')] for l in lines[:num_examples]]
    print(wordpairs[:5])
    return zip(*wordpairs)

en, sp = create_dataset(path_to_file, None)
print(en[-1])
print(sp[-1])

def max_length(tensor):
    return max(len(t) for t in tensor)

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    
    return tensor, lang_tokenizer

# This is a work in progress. The objective here is to train a model to translate spanish <-> English.
# Let's Go!