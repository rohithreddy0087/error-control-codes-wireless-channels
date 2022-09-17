# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 22:32:26 2021

@author: Dell
"""
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from scipy.special import erfc #erfc/Q function
import pandas as pd  
import torch,torchvision
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from model import Encoder,Decoder
from dataset import generate_dataset
from channel import noise
import os 

def bool2int(x):
    p = []
    for i in x :
        c = int(i[0])*8+int(i[1])*4+int(i[2])*2+int(i[3])*1
        p.append(c)
    return np.array(p)

def int2bin(i):
    m = [int(x) for x in format(i, 'b').zfill(4)]
    msg = np.array(m)
    return msg
h = np.array([
    [0,0,0,0,0,0,0],
    [0,0,0,1,0,1,1],
    [0,0,1,0,1,0,1],
    [0,0,1,1,1,1,0],
    [0,1,0,0,1,1,0],
    [0,1,0,1,1,0,1],
    [0,1,1,0,0,1,1],
    [0,1,1,1,0,0,0],
    [1,0,0,0,1,1,1],
    [1,0,0,1,1,0,0],
    [1,0,1,0,0,1,0],
    [1,0,1,1,0,0,1],
    [1,1,0,0,0,0,1],
    [1,1,0,1,0,1,0],
    [1,1,1,0,1,0,0],
    [1,1,1,1,1,1,1]
    ])
h = 2*h-1


nSym = int(10e5) 
snrdB = np.arange(0,8)
snr_lin = 10**(snrdB/10)
EbN0dBs = 10*np.log10(snr_lin*7/4);
BER_sim = np.zeros(len(EbN0dBs)) 
errors = np.zeros(len(EbN0dBs)) 
M=2 
#[0.0107,0.00459025,0.001600375,0.0004345,8.05e-05,8.83333333333333e-06]
k = int(nSym/4)
for j,snr in enumerate(snr_lin):
    count = 0
    while errors[j] < 400:
        inputSyms = np.random.randint(low=0, high = M, size=nSym) 
        inputSyms = 2*inputSyms - 1
        inputSyms = inputSyms.reshape(-1,4)
        inp= torch.Tensor(inputSyms)
        #s = torch.Tensor(snr)
        msg1 = (inp+1)/2
        msg1 = bool2int(msg1)
        encoded = torch.Tensor(h[msg1])
        Es = torch.mean(torch.mean(encoded**2,axis = 1))
        
        noise_power = Es/snr
        sigma = torch.sqrt(noise_power/2)
        n = sigma*(torch.randn(250000, 7))#+1j*torch.randn(1000000, 7))
        r = n + encoded
        r = r.numpy()
        r = r.T
        for i in range(r.shape[1]):
            temp_r = r[:,i].reshape((7,1))
            diff = np.kron(np.ones((1,16)),temp_r)-h.T
            min_c = np.argmin(np.sum(diff*np.conj(diff),0))
            
            y1 = int2bin(min_c)
            y1 = 2*y1-1
            errors[j] = errors[j] + np.sum(y1!= inputSyms[i,:])
        #print(errors[j])
        count = count + 1
    BER_sim[j] = errors[j]/(2*nSym*count) 
    print(EbN0dBs[j],BER_sim[j])

# BER_sim = [7.5e-02,5.2e-02, 3e-02, 1.8e-02, 0.9e-02, 4.7e-03, 1.8e-03, 6.8e-04,
#        1e-04, 2.6e-05,3e-06]


dict = {'EbN0 (in dBs)': EbN0dBs, 'BER': BER_sim}  
     
df = pd.DataFrame(dict) 
  
# saving the dataframe 
df.to_csv('hamming_ber.csv') 


