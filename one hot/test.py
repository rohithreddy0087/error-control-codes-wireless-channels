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
def convertToOneHot(vector, num_classes=None):
    result = np.zeros(shape=(1,num_classes))
    result[np.arange(1), vector] = 1
    return result.astype(int)

model = Autoencoder()
checkpoint = tf.train.Checkpoint(model)

checkpoint.restore('ckpts-1/test_model/ckpt-1')
EbN0dB = 8
snr = 10**((EbN0dB/10)*(4/7))
testset_x = np.zeros((16,17))

k = 0
for j in range(16):
    testset_x[j][0:16] = convertToOneHot(k, num_classes=16)
    testset_x[j][16:17] = snr#[j]
    k = k + 1
    if k == 16:
        k = 0



d,en = model.predict(testset_x)

enc = en >= 0
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