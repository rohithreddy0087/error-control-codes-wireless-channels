# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:27:39 2021

@author: Dell
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses

from tensorflow.keras.models import Model

from model import Autoencoder,Encoder,Decoder
from dataset import generate_dataset
from channel import noise
np.set_printoptions(precision=2)

def ReLU(x):
    return x * (x > 0)

def layer(inp,W,b):
    z = np.matmul(W.T,inp.T) + b
    return ReLU(z)

model_encoder = Encoder(err_dim=7)
checkpoint = tf.train.Checkpoint(model_encoder)

checkpoint.restore('ckpts-1/model_encoder_only1/ckpt-1')

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
# for i in range(16):
#   print(msg[i],en[i])
# d1 = np.loadtxt('wts/4_32.txt')
# b1 = np.loadtxt('wts/32_2.txt')
# d2 = np.loadtxt('wts/32_16.txt')
# b2 = np.loadtxt('wts/16_4.txt')
# d3 = np.loadtxt('wts/16_7.txt')
# b3 = np.loadtxt('wts/7_6.txt')

d1 = np.loadtxt('new_wts/weight1.txt')
b1 = np.loadtxt('new_wts/bias1.txt')
d2 = np.loadtxt('new_wts/weight2.txt')
b2 = np.loadtxt('new_wts/bias2.txt')
d3 = np.loadtxt('new_wts/weight3.txt')
b3 = np.loadtxt('new_wts/bias3.txt')
d4 = np.loadtxt('new_wts/weight4.txt')
b4 = np.loadtxt('new_wts/bias4.txt')
d5 = np.loadtxt('new_wts/weight5.txt')
b5 = np.loadtxt('new_wts/bias5.txt')
# d1 = d1+ np.random.normal(0,0.1,(512,4))
# b1 = b1+ np.random.normal(0,0.1,(512,))

# d2 = d2+ np.random.normal(0,0.1,(128,512))
# b2 = b2+ np.random.normal(0,0.1,(128,))

# d3 = d3+ np.random.normal(0,0.1,(64,128))
# b3 = b3+ np.random.normal(0,0.1,(64,))

# d4 = d4+ np.random.normal(0,0.1,(32,64))
# b4 = b4+ np.random.normal(0,0.1,(32,))

# d5 = d5+ np.random.normal(0,0.1,(7,32))
# b5 = b5+ np.random.normal(0,0.1,(7,))

lst = []
lst.append(d1.T)
lst.append(b1)
lst.append(d2.T)
lst.append(b2)
lst.append(d3.T)
lst.append(b3)
lst.append(d4.T)
lst.append(b4)
lst.append(d5.T)
lst.append(b5)
model_encoder.set_weights(lst)
en1 = model_encoder.predict(testset_x)
for i in range(16):
  print(msg[i],en1[i])

checkpoint.save('ckpts-1/model_encoder_wts')
#enc = en1
L = []
D = np.zeros((16,16))
C= np.transpose(en1)
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
print("Minimum eucledian distance is "+str(np.min(D)))

def hamming_distance(chaine1, chaine2):
    return sum(c1 != c2 for c1, c2 in zip(chaine1, chaine2))

enc = en1 >= 0
enc = 2*enc-1
enc = enc.astype(np.float32)
ham_dist = np.zeros_like(D)
for i in range(16):
    for j in range(16):
        ham_dist[i][j]  = hamming_distance(enc[i],enc[j])
diag = np.ones((16,))*10
np.fill_diagonal(ham_dist,diag)
print("Minimum hamming distance is "+str(np.min(ham_dist)))
