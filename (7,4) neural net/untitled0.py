# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 22:32:26 2021

@author: Dell
"""
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from scipy.special import erfc #erfc/Q function
from model import Autoencoder
import tensorflow as tf


model = Autoencoder()
checkpoint = tf.train.Checkpoint(model)

checkpoint.restore('ckpts/model-2')
nSym = 10**3 
EbN0dBs = np.arange(start=1,stop = 20, step = 1) 
BER_sim = np.zeros(len(EbN0dBs)) 

M=2 


inputS = np.random.randint(low=0, high = M, size=nSym) 
inputSyms =inputS.reshape(-1,4)

for j,EbN0dB in enumerate(EbN0dBs):
    print(j)
    gamma = 10**(EbN0dB/10) 
    s = np.ones((250000,1))*gamma
    inp = np.concatenate((inputSyms,s),axis = 1)

    c,_ = model.predict(inp)

    r = np.round(c)
    
    BER_sim[j] = np.sum(r != inputSyms)/nSym 

BER_theory = 0.5*erfc(np.sqrt(10**(EbN0dBs/10)))

fig, ax = plt.subplots(nrows=1,ncols = 1)
ax.semilogy(EbN0dBs,BER_sim,color='r',marker='o',linestyle='',label='BPSK Sim')
ax.semilogy(EbN0dBs,BER_theory,marker='',linestyle='-',label='BPSK Theory')
ax.set_xlabel('$E_b/N_0(dB)$');ax.set_ylabel('BER ($P_b$)')
ax.set_title('Probability of Bit Error for BPSK over AWGN channel')
ax.set_xlim(1,20);ax.grid(True);
ax.legend();plt.show()

plt.figure()
plt.plot(EbN0dBs,BER_sim,color='r',marker='o',linestyle='',label='BPSK Sim')