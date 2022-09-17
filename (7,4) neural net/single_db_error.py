# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 21:15:40 2021

@author: Dell
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses

from tensorflow.keras.models import Model

from model import Autoencoder
from dataset import generate_dataset

model = Autoencoder()
checkpoint = tf.train.Checkpoint(model)

checkpoint.restore('ckpts/train_model/ckpt-1')
nSym = 10**4

M=2 


#------------ Transmitter---------------
inputSyms = np.random.randint(low=0, high = M, size=nSym)
msg = np.zeros((1,5))
gamma = 10**0.8
out = []

for k in range(int(len(inputSyms)/4)):
    inp = inputSyms[4*k:4*(k+1)]
    msg[:,0:4] = inp
    msg[:,4] = gamma
    c,_ = model.predict(msg)
    out.append(c)
    ot = np.array(out)
    ot = np.round(ot)
r = ot.flatten()

#-------------- Receiver ------------
print(np.sum(r != inputSyms)/nSym)