#long short tern memory NN generate city names
from __future__ import absolute_import, division, print_function
import sys
sys.path.append(__file__)
import os
from six import moves
import ssl
import tensorflow as tf
import tflearn
from tflearn.data_utils import *
import ipdb

#Step1---Retrive the data
path="US_cities.txt"
#if not os.path.isfile(path):
    #context = ssl._create_unverified_context()
    #moves.urllib.request.urlretrieve("https:raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/US_cities", path, context=context)

#city name max length
maxlen = 20

#vectorize the text file
X, Y, char_idx = textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3)  #X-input ,Y-target, char_idx--dictionary

ipdb.set_trace()
#creat LSTM
g = tflearn.input_data(shape=[None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 512, return_seq = True)   #512 nodes in this layer
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(char_idx), activation="softmax")

#generate names
m = tflearn.SequenceGenerator(g, dictionary=char_idx,seq_maxlen=maxlen, checkpoint_path="model_us_cities")#clip_gradients=5.0,


#training
for i in range(40):
    seed = random_sequence_from_textfile(path, maxlen)
    m.fit(X, Y, validation_set=0.1, batch_size=128, n_epoch=1, run_id="us cities") #validation_set=0.1: 10% of training data used for validation

    print("Testing")
    print(m.generate(30, temperature=0.5, seq_seed=seed))
    print("Testing")
    print(m.generate(30, temperature=1.2, seq_seed=seed))
    print("Testing")
    print(m.generate(30, temperature=1.0, seq_seed=seed))
    print("Testing")
    print(m.generate(30, temperature=1.2, seq_seed=seed))
    print("Testing")
    print(m.generate(30, temperature=1.5, seq_seed=seed))
