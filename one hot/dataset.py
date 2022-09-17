# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 19:27:25 2021

@author: Dell
"""
import random
import numpy as np
import tensorflow as tf
def convertToOneHot(vector, num_classes=None):
    result = np.zeros(shape=(1,num_classes))
    result[np.arange(1), vector] = 1
    return result.astype(int)

def sample_floats(low, high, k=1):
    """ Return a k-length list of unique random floats
        in the range of low <= x <= high
    """
    result = []
    seen = set()
    for i in range(k):
        x = random.uniform(low, high)
        while x in seen:
            x = random.uniform(low, high)
        seen.add(x)
        result.append(x)
    return result

min_snr = 1
max_snr = 100

def generate_dataset(length = 1):
    msg = []
    for i in range(16):
        m = [int(x) for x in format(i, 'b').zfill(4)]
        msg.append(m)
        
    #snr = sample_floats(min_snr,max_snr,length)
    EbN0dB = np.linspace(1,20,length)
    EbN0dB = 8
    snr = 10**((EbN0dB/10)*(4/7))
    #snr = 10**(8/10)
    dataset_x = np.zeros((length*16,17))
    dataset_y = np.zeros((length*16,16))
    t = 0
    for j in range(length):
        for k in range(len(msg)):
            dataset_x[t][0:16] = convertToOneHot(k, num_classes=16)
            dataset_x[t][16:17] = snr#[j]
            dataset_y[t][0:16] = convertToOneHot(k, num_classes=16)
            t = t + 1
    #snr = sample_floats(min_snr,max_snr,length*4) 
    testset_x = np.zeros((length*4,17))
    testset_y = np.zeros((length*4,16))  
    
    k = 0
    for j in range(length*4):
        testset_x[j][0:16] = convertToOneHot(k, num_classes=16)
        testset_x[j][16:17] = snr#[j]
        testset_y[j][0:16] = convertToOneHot(k, num_classes=16)
        k = k + 1
        if k == 16:
            k = 0
            
    return dataset_x,dataset_y,testset_x,testset_y