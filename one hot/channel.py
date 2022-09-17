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