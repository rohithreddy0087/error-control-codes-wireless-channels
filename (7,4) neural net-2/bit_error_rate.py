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
import pandas as pd  
pd.set_eng_float_format(accuracy=1, use_eng_prefix=False)

model = Autoencoder()
checkpoint = tf.train.Checkpoint(model)

checkpoint.restore('ckpts-1/train_model/ckpt-1')
nSym = 10**6 
EbN0dBs = np.arange(start=1,stop = 20, step = 1) 
BER_sim = np.zeros(len(EbN0dBs)) 

M=2 
inputSyms = np.random.randint(low=0, high = M, size=nSym) #Random 1's and 0's as input to BPSK modulator
inputSyms =inputSyms.reshape(-1,4)
k = int(nSym/4)
for j,EbN0dB in enumerate(EbN0dBs):
    print(j)
    gamma = 10**((EbN0dB/10)*(4/7))
    s = np.ones((k,1))*gamma
    inp = np.concatenate((inputSyms,s),axis = 1)

    c,_ = model.predict(inp)

    r = np.round(c)
    BER_sim[j] = np.sum(r != inputSyms)/nSym 

BER_theory = 0.5*erfc(np.sqrt(10**(EbN0dBs/10)))

dict = {'EbN0 (in dBs)': EbN0dBs, 'BER': BER_sim}  
     
df = pd.DataFrame(dict) 
  
# saving the dataframe 
df.to_csv('model-11-6.csv') 

fig, ax = plt.subplots(nrows=1,ncols = 1)
ax.semilogy(EbN0dBs,BER_sim,color='r',marker='o',linestyle='',label='BPSK Sim')
ax.semilogy(EbN0dBs,BER_theory,marker='',linestyle='-',label='BPSK Theory')
ax.set_xlabel('$E_b/N_0(dB)$');ax.set_ylabel('BER ($P_b$)')
ax.set_title('Probability of Bit Error for BPSK over AWGN channel')
ax.set_xlim(1,20);ax.grid(True);
ax.legend();plt.show()
fig.savefig('fig-ecc-11-1-6')
plt.figure()
plt.plot(EbN0dBs,BER_sim,color='r',marker='o',linestyle='',label='BPSK Sim')
plt.savefig('fig-ecc-11-2-6')
