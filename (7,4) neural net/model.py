# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 00:22:03 2021

@author: Dell
"""
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model

from channel import noise,channel1

class Encoder(Model):
    def __init__(self,err_dim):
        super(Encoder, self).__init__()
        self.encode = tf.keras.Sequential([
                          layers.Dense(512, activation='relu'),
                          layers.BatchNormalization(),
                          layers.Dense(128, activation='relu'),
                          layers.BatchNormalization(),
                          layers.Dense(64, activation='relu'),
                          layers.BatchNormalization(),
                          layers.Dense(32, activation='relu'),
                          layers.BatchNormalization(),
                          layers.Dense(7, activation='tanh'),
                          #layers.BatchNormalization(),
                          #layers.Activation('tanh'),
                          #layers.LayerNormalization(),
                        ])
    def call(self, x):
        encoded = self.encode(x)
        return encoded

class Decoder(Model):
    def __init__(self,code_dim):
        super(Decoder, self).__init__()
        self.decode = tf.keras.Sequential([
                          layers.Dense(16,input_shape = (7,), activation='tanh'),
                          layers.Dense(16,input_shape = (7,), activation='tanh'),
                          layers.Dense(8,input_shape = (7,), activation='tanh'),
                          layers.Dense(code_dim, activation='sigmoid')
                        ])

    def call(self, x):
        decoded = self.decode(x)
        return decoded

class Autoencoder(Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = Encoder(err_dim=7)
    self.decoder = Decoder(code_dim=4)
  def call(self,msg):
    x = msg[:,0:4]
    #x = tf.cast(x,tf.int32)
    encoded = self.encoder(x)
    #Es = tf.math.reduce_mean(encoded**2,axis = 1)
    #Es = tf.ones_like(Es)
    #n = noise(msg[:,4:5],Es)
    #r = tf.math.add(encoded,n)
    r = channel1(encoded)
    decoded = self.decoder(r)
    return decoded,encoded

# inp =  np.array([[1,0,1,1,8.4],[1,1,1,1,2.6]])

# a = Autoencoder()
# e = a(inp)