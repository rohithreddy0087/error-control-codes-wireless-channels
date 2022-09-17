# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 00:22:03 2021

@author: Dell
"""
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from channel import noise

class Encoder(Model):
    def __init__(self,err_dim):
        super(Encoder, self).__init__()
        self.encode = tf.keras.Sequential([
                          layers.Dense(16,input_shape = (16,), activation='relu'),
                          layers.Dense(err_dim, activation='linear'),
                          layers.LayerNormalization()
                        ])
    def call(self, x):
        encoded = self.encode(x)
        return encoded

class Decoder(Model):
    def __init__(self,code_dim):
        super(Decoder, self).__init__()
        self.decode = tf.keras.Sequential([
                          layers.Dense(code_dim,input_shape = (7,), activation='linear'),
                          layers.Dense(code_dim, activation='relu'),
                          layers.Softmax(axis=-1)
                        ])

    def call(self, x):
        decoded = self.decode(x)
        return decoded



class Autoencoder(Model):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder = Encoder(err_dim=7)
    self.decoder = Decoder(code_dim=16)
  def call(self,msg):
    x = msg[:,0:16]
    x = tf.cast(x,tf.int32)
    encoded = self.encoder(x)
    Es = tf.math.reduce_mean(encoded**2,axis = 1)
    Es = tf.ones_like(Es)
    n = noise(msg[:,16:17],Es)
    r = tf.math.add(encoded,n)
    decoded = self.decoder(r)
    return decoded,encoded

# inp =  np.array([[1,0,1,1,8.4],[1,1,1,1,2.6]])

# a = Autoencoder()
# e = a(inp)