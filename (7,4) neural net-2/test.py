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

from model import Autoencoder
np.set_printoptions(precision=1)
model = Autoencoder()
checkpoint = tf.train.Checkpoint(model)

checkpoint.restore('ckpts-1/train_model/ckpt-1')

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
    
testset_x = np.zeros((16,5))
for j in range(16):
    testset_x[j][0:4] = msg[j]
    testset_x[j][4:5] = snr
inp = np.array([1,1,1,1,snr]).reshape((1,5))
inp = inp.astype(np.float64)

y_pred,en = model.predict(testset_x)
for i in range(16):
  print(msg[i],en[i])
