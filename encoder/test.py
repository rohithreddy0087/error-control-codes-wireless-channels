# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 23:12:38 2021

@author: Dell
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses

from tensorflow.keras.models import Model,model_from_json

from model import Autoencoder,Encoder,Decoder
np.set_printoptions(precision=1)
model_encoder = Encoder(err_dim=7)
checkpoint = tf.train.Checkpoint(model_encoder)

checkpoint.restore('ckpts-1/model_encoder_only1/ckpt-1')

# json_file = open('model.json','r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)

# loaded_model.load_weights('1st_model.h5')
snr = 10**(11/10)
msg = []
for i in range(16):
    m = [int(x) for x in format(i, 'b').zfill(4)]
    msg.append(m)
    
testset_x = np.zeros((16,4))
for j in range(16):
    testset_x[j][0:4] = msg[j]
    #testset_x[j][4:5] = snr

testset_x = 2*testset_x-1

en = model_encoder.predict(testset_x)
for i in range(16):
  print(msg[i],en[i])

enc = en >= 0.5
enc = 2*enc/2.0
enc = enc.astype(np.float32)
#enc = en
L = []
D = np.zeros((16,16))
C= np.transpose(enc)
n = tf.ones((16,))
for i in range(16):
    ci = np.reshape(C[:,i],(1,7))
    cj = np.transpose(ci)
    norm = np.reshape(np.matmul(ci,cj),(1,))
    L.append(norm*n)
L = np.stack(L)
D = L-2*np.matmul(np.transpose(C),C)+np.transpose(L)
diag = np.ones((16,))*10
np.fill_diagonal(D,diag)
print("Minimum hamming distance is "+str(np.min(D)))