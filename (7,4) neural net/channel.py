# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 13:06:55 2021

@author: Dell
"""
import numpy as np
import tensorflow as tf
def noise(snr,Es):
    var = tf.reshape(Es/tf.reshape(snr,(-1,)),(-1,1))
    n = tf.random.normal([1,7],0,var[:])
    return n
def get_sigma(snr):
    return 10 ** (-snr * 1.0/20)
def channel1(x):
        #print('training with noise snr db', args.train_channel_low, args.train_channel_high)
    train_channel_low = 0
    train_channel_high = 10
    noise_sigma_low =  get_sigma(train_channel_low) # 0dB
    noise_sigma_high =  get_sigma(train_channel_high) # 0dB
    #print('training with noise snr db', noise_sigma_low, noise_sigma_high)
    noise_sigma =  tf.random.uniform(tf.shape(x),
        minval=noise_sigma_high,
        maxval=noise_sigma_low,
        dtype=tf.float32
    )

    return x+ noise_sigma*tf.random.normal(tf.shape(x),dtype=tf.float32, mean=0., stddev=1.0) 