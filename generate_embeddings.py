# Python program to generate embedding and save in a file
# Author- Manish Patel
# Date- 09 April 2019
# Input -Quora Question Pairs

import os
data_file='quora_duplicate_questions.tsv'
# 0 means dont load, 1 means fetch from file
LOAD_ENCODING_FROM_FILE=0 
encoding_data_file_quest1='encoding_quest1'
encoding_data_file_quest2='encoding_quest2'
encoding_data_file_label='label'

#################################################
import numpy as np
import pandas as pd
import tensorflow as tf
import re
from bert_serving.client import BertClient
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle




#################################################
#Helper functions
def sanitize_question(text):
	''' Pre process and convert texts to a list of words 
	'''
	text = str(text)
	text = text.lower()
# Clean the text
	text = re.sub(r"[^A-Za-z0–9^,!.\/'+-=]", " ", text)
	text = re.sub(r"what's", "what is ", text)
	text = re.sub(r"\'s", " ", text)
	text = re.sub(r"\'ve", " have ", text)
	text = re.sub(r"can't", "cannot ", text)
	text = re.sub(r"n't", " not ", text)
	text = re.sub(r"i'm", "i am ", text)
	text = re.sub(r"\'re", " are ", text)
	text = re.sub(r"\'d", " would ", text)
	text = re.sub(r"\'ll", " will ", text)
	text = re.sub(r",", " ", text)
	text = re.sub(r"\.", " ", text)
	text = re.sub(r"!", " ! ", text)
	text = re.sub(r"\/", " ", text)
	text = re.sub(r"\^", " ^ ", text)
	text = re.sub(r"\+", " + ", text)
	text = re.sub(r"\-", " - ", text)
	text = re.sub(r"\=", " = ", text)
	text = re.sub(r"'", " ", text)
	text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
	text = re.sub(r":", " : ", text)
	text = re.sub(r" e g ", " eg ", text)
	text = re.sub(r" b g ", " bg ", text)
	text = re.sub(r" u s ", " american ", text)
	text = re.sub(r"\0s", "0", text)
	text = re.sub(r" 9 11 ", "911", text)
	text = re.sub(r"e - mail", "email", text)
	text = re.sub(r"j k", "jk", text)
	text = re.sub(r"\s{2,}", " ", text)
	#text = text.split()
	return text

#################################################
# Read the input file
data_df=pd.read_csv(data_file,sep='\t',nrows=100000)
print(data_df.shape)

sent1=[]
sent2=[]
label=[]
for index, row in data_df.iterrows():
	sent1.append(sanitize_question(row[3]))
	sent2.append(sanitize_question(row[4]))
	label.append(sanitize_question(row[5]))


maxlen = 125  # We will cut reviews after 125 words
#training_samples = 300  # We will be training on 300 samples
#validation_samples = 200  # We will be validating on 200 samples
#max_words = 10000  # We will only consider the top 10,000 words in the dataset

# The next step is to tranform all sentences to fixed length encoding using bert embeddings
# [0.1 0.4 0.4] [0.9 0.6 0.1] 2.4
# [0.4 0.1 0.3] [0.5 0.6 0.1] 1.0

# Save the encodings in a file 
if LOAD_ENCODING_FROM_FILE == 0:
	bc=BertClient(port=5555, port_out=5556)
	vec1=bc.encode(sent1)
	with open(encoding_data_file_quest1, "wb") as fp:
		pickle.dump(vec1, fp)
	vec2=bc.encode(sent2)
	with open(encoding_data_file_quest2, "wb") as fp:   
		pickle.dump(vec2,fp)
	with open(encoding_data_file_label, "wb") as fp: 
		pickle.dump(label,fp)
exit(0)

