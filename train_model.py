# Python program to generate embedding and save in a file
# Author- Manish Patel
# Date- 09 April 2019
# Input -Quora Question Pairs

import os
data_file='/home/manishp/Research/quora_duplicate_questions.tsv'
# 0 means dont load, 1 means fetch from file
LOAD_ENCODING_FROM_FILE=1 
encoding_data_file_quest1='/home/manishp/Research/encoding_quest1'
encoding_data_file_quest2='/home/manishp/Research/encoding_quest2'
encoding_data_file_label='/home/manishp/Research/quest_label'

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
from keras import models
from keras import layers
from keras import optimizers
from keras.layers import Dropout

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


maxlen = 125  # We will cut reviews after 125 words
#training_samples = 300  # We will be training on 300 samples
#validation_samples = 200  # We will be validating on 200 samples
#max_words = 10000  # We will only consider the top 10,000 words in the dataset

# The next step is to tranform all sentences to fixed length encoding using bert embeddings
# [0.1 0.4 0.4] [0.9 0.6 0.1] 2.4
# [0.4 0.1 0.3] [0.5 0.6 0.1] 1.0

# Save the encodings in a file 
if LOAD_ENCODING_FROM_FILE == 1:
	#bc=BertClient(port=5555, port_out=5556)
	#vec1=bc.encode(sent1)
	with open(encoding_data_file_quest1, "rb") as fp:
		vec1=pickle.load(fp)
	#vec2=bc.encode(sent2)
	with open(encoding_data_file_quest2, "rb") as fp:   
		vec2=pickle.load(fp)
	with open(encoding_data_file_label, "rb") as fp: 
		label=pickle.load(fp)

#label=label[:500]
train_vec1 = np.asarray(vec1, np.float32)
train_vec2 = np.asarray(vec2, np.float32)
train_label = np.asarray(label,np.float32)
print(np.shape(train_vec1))
print(np.shape(train_vec2))
print(np.shape(train_label))

#vec1 = tf.convert_to_tensor(vec1, np.float32)
#vec2 = tf.convert_to_tensor(vec2, np.float32)
#label= tf.convert_to_tensor(label,np.float32)

# vec1 shape is not correct
import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model


inp1= Input(shape=(768,))
inp2= Input(shape=(768,))

x = keras.layers.concatenate([inp1, inp2],axis=-1)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5) (x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5) (x)
x = Dense(64, activation='relu')(x)
out=Dense(1, activation='sigmoid')(x)
model = Model(inputs=[inp1,inp2], outputs=out)
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history=model.fit([train_vec1, train_vec2], train_label, 
	epochs=30,batch_size=200,
	validation_split=0.3)

#################################################
# Save model
# First model uses mean_squared_error, 256, 0.5, 64, 1
# second model uses binary_crossentropy,256,0.5,64,1
#with sigmoid 
model.save('second_model.h5')
#################################################
# Plot figures
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#################################################
