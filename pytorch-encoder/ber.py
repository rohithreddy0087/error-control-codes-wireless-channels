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

def int2bin(x):
    msg = []
    for i in x:
        m = [int(x) for x in format(i, 'b').zfill(4)]
        msg.append(m)
    msg = np.array(msg)
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

np.set_printoptions(precision=1)
model_encoder = Encoder()
model_decoder = Decoder()

model_encoder.load_state_dict(torch.load('ckpts/model_encoder_only/weights', map_location=lambda storage, loc: storage))
model_decoder.load_state_dict(torch.load('ckpts/model_decoder_DL/weights', map_location=lambda storage, loc: storage))

nSym = int(10e5) 
snrdB = np.arange(0,8)
snr_lin = 10**(snrdB/10)
EbN0dBs = 10*np.log10(snr_lin*7/4);
BER_sim = np.zeros(len(EbN0dBs)) 

M=2 
#[0.0107,0.00459025,0.001600375,0.0004345,8.05e-05,8.83333333333333e-06]
k = int(nSym/4)
for j,snr in enumerate(snr_lin):
    inputSyms = np.random.randint(low=0, high = M, size=nSym) 
    inputSyms = 2*inputSyms - 1
    inputSyms = inputSyms.reshape(-1,4)
    inp= torch.Tensor(inputSyms) 
    #encoded = model_encoder(inp)
    msg1 = (inp+1)/2
    msg1 = bool2int(msg1)
    encoded = torch.Tensor(h[msg1])
    Es = torch.mean(torch.mean(encoded**2,axis = 1))
        
    noise_power = Es/snr
    sigma = torch.sqrt(noise_power/2)
    n = sigma*(torch.randn(250000, 7))#+1j*torch.randn(1000000, 7))
    r = n + encoded
    decoded = model_decoder(r)
    decode = decoded > 0
    y1 = 2*decode-1
    # y = torch.argmax(decoded, dim=1)
    # y1 = int2bin(y.numpy())
    # y1 = 2*y1-1
    BER_sim[j] = np.sum(y1.numpy()!= inp.numpy())/(2*nSym) 
    print(EbN0dBs[j], BER_sim[j])

BER_theory = 0.5*erfc(np.sqrt(10**(EbN0dBs/10)))
# BER_sim = [7.5e-02,5.2e-02, 3e-02, 1.8e-02, 0.9e-02, 4.7e-03, 1.8e-03, 6.8e-04,
#        1e-04, 2.6e-05,3e-06]


dict1 = {'EbN0 (in dBs)': EbN0dBs, 'BER': BER_sim}  
     
df = pd.DataFrame(dict1) 
  
# saving the dataframe 
df.to_csv('model_sim_DL.csv') 


