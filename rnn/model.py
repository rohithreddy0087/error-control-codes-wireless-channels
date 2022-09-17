# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 16:52:47 2021

@author: Dell
"""
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tf.keras.layers import Input, Dense, Flatten, Reshape, GaussianNoise, Activation, Subtract
from tf.keras.layers import LSTM, GRU, SimpleRNN, TimeDistributed, Multiply, Add
from tf.keras.layers.normalization import BatchNormalization
from tf.keras.layers.wrappers import  Bidirectional
from tf.keras.models import Model
import tf.keras.backend as K

def create_gru_model(coderate, input_dim, sigma):
    # step activation
    def bpsk_activation(x):
        return K.sign(x - 0.5)
    
    def scale_activation(x):
        return 2 * (x - 0.5)
    
    enc_gru_units = 25
    dec_gru_units = 100
    
    x = Input(shape=(input_dim, 1))
    gru_enc1 = GRU(name='enc_gru_1', units=enc_gru_units, activation='tanh', return_sequences=True, dropout=1.0)
    gru_enc2 = GRU(name='enc_gru_2', units=enc_gru_units, activation='tanh', return_sequences=True, dropout=1.0)
    output_enc = Dense(coderate, activation='sigmoid') # try timedistributed
    
    encoded = output_enc(gru_enc2(gru_enc1(x)))
    # BPSK
    # encoded = Activation(bpsk_activation)(encoded)
    encoded = Activation(scale_activation)(encoded)
    # Noise
    input_noise = Input(shape=(input_dim, coderate))
    # noise = GaussianNoise(sigma)(K.zeros(shape=(input_dim, coderate)), training=True)
    # noised = Add()([input_noise, noise])
    noised = GaussianNoise(sigma)(input_noise, training=True)
    
    y = Input(shape=(input_dim, coderate))
    gru_dec1 = GRU(name='dec_gru_1', units=dec_gru_units, activation='tanh', return_sequences=True, dropout=1.0)
    gru_dec2 = GRU(name='dec_gru_2', units=dec_gru_units, activation='tanh', return_sequences=True, dropout=1.0)
    gru_dec3 = GRU(name='dec_gru_3', units=dec_gru_units, activation='tanh', return_sequences=True, dropout=1.0)
    gru_dec4 = GRU(name='dec_gru_4', units=dec_gru_units, activation='tanh', return_sequences=True, dropout=1.0)
    output_dec = Dense(1, activation='sigmoid')
    
    decoded = output_dec(gru_dec4(gru_dec3(gru_dec2(gru_dec1(y)))))
    
    # Модели
    encoder = Model(x, encoded, name="encoder")
    #encoder.summary()
    noise = Model(input_noise, noised, name = "noise");
    #noise.summary()
    decoder = Model(y, decoded, name="decoder")
    #decoder.summary()
    autoencoder = Model(x, decoder(noise(encoder(x))), name="autoencoder")
    return encoder, decoder, autoencoder, noise