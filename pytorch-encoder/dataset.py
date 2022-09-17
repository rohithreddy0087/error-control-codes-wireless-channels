# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 19:27:25 2021

@author: Dell
"""
import random
import numpy as np
from itertools import combinations
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
    n = 4
    b = 7
    k = 2**n
    for i in range(k):
        m = [int(x) for x in format(i, 'b').zfill(n)]
        msg.append(m)
    #snr = sample_floats(min_snr,max_snr,length)
    snrdB = np.arange(0,10)
    snrdB = 0
    snr = 10**(snrdB/10)
    #snr = 10**(8/10)
    dataset_x = np.zeros((length*k,n+1))
    dataset_y = np.zeros((length*k,n))
    t = 0
    for j in range(length):
        for i in range(len(msg)):
            dataset_x[t][0:n] = msg[i]
            dataset_x[t][n:n+1] = snr#[j]
            dataset_y[t][0:n] = msg[i]
            t = t + 1
    snr = sample_floats(min_snr,max_snr,length*n) 
    testset_x = np.zeros((length*n,n+1))
    testset_y = np.zeros((length*n,n))  
    
    k1 = 0
    for j in range(length*n):
        testset_x[j][0:n] = msg[k1]
        testset_x[j][n:n+1] = snr[j]
        testset_y[j][0:n] = msg[k1]
        k1 = k1 + 1
        if k1 == k:
            k1 = 0
            
    return dataset_x,dataset_y,testset_x,testset_y

def make_datasets():
    msg = []
    n = 4
    b = 7
    k = 2**n
    for i in range(k):
        m = [int(x) for x in format(i, 'b').zfill(n)]
        msg.append(m)
    
    num = np.arange(0,16,1)
    comb = combinations(num, 4) 
    data = []
    for i in list(comb):
        for j in range(len(i)):
            data.append(msg[i[j]])
    data = np.array(data)
    return data
