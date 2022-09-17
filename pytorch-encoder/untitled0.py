# -*- coding: utf-8 -*-
"""
Created on Wed May 19 19:23:07 2021

@author: Dell
"""
import matplotlib.pyplot as plt
import numpy as np

snrdB = np.arange(0,7)
snr_lin = 10**(snrdB/10)
EbN0dBs1 = 10*np.log10(snr_lin*7/4);

snrdB = np.arange(0,30)
snr_lin = 10**(snrdB/10)
EbN0dBs2 = 10*np.log10(snr_lin*7/4);

shd = [0.0107,0.00459025,0.001600375,0.0004345,8.05e-05,8.83333333333333e-06,1e-06]
shd_f = [0.056967  , 0.04735   , 0.039714  , 0.032279  , 0.0265725 ,
       0.0218405 , 0.0171645 , 0.0139705 , 0.011615  , 0.0091185 ,
       0.007389  , 0.005867  , 0.004713  , 0.0036915 , 0.0030015 ,
       0.0023985 , 0.00186   , 0.0014685 , 0.001189  , 0.000982  ,
       0.0007865 , 0.0006325 , 0.000473  , 0.0003915 , 0.0003    ,
       0.0002205 , 0.0001995 , 0.00015325, 0.0001265 , 0.00010025]
fig, ax = plt.subplots(nrows=1,ncols = 2)
fig.set_figheight(20)
fig.set_figwidth(30)
ax[0].semilogy(EbN0dBs1,shd,color='r',marker='o',linestyle='-',label='Hamming encoder-soft decision decoder')
ax[1].semilogy(EbN0dBs2,shd_f,color='b',marker='d',linestyle='-',label='Hamming encoder-soft decision decoder')
#ax[1].semilogy(EbN0dBs1,d_7_4,color='r',marker='o',linestyle='-',label='(7,4) Auto-encoder')

# ax.semilogy(EbN0dBs,d_7_4,color='g',marker='v',linestyle='-',label='7,4')
# EbN0dBs = 10*np.log10(snr_lin*15/11)
# ax.semilogy(EbN0dBs,d_15_11,color='y',marker='v',linestyle='-',label='15,11')
# EbN0dBs = 10*np.log10(snr_lin*21/11)
# ax.semilogy(EbN0dBs,d_21_11,color='brown',marker='v',linestyle='-',label='21,11')

ax[0].legend(prop={"size":24})
ax[0].set_xlabel('$SNR(dB)$',fontsize=24)
ax[0].set_ylabel('BER',fontsize=24)
ax[0].set_title('BER vs SNR curve in AWGN channel',fontsize=30)
for tick in ax[0].xaxis.get_major_ticks():
    tick.label.set_fontsize(20)
for tick in ax[0].yaxis.get_major_ticks():
    tick.label.set_fontsize(20)
#ax.set_xlim(2.2,10);ax.set_ylim(10e-7,0.05)
ax[0].grid(True)
ax[1].legend(prop={"size":24})
ax[1].set_xlabel('$SNR(dB)$',fontsize=24)
ax[1].set_ylabel('BER',fontsize=24)
for tick in ax[1].xaxis.get_major_ticks():
    tick.label.set_fontsize(20)
for tick in ax[1].yaxis.get_major_ticks():
    tick.label.set_fontsize(20)
ax[1].set_title('BER vs SNR curve in Fading channel',fontsize=30)
#ax.set_xlim(2.2,10);ax.set_ylim(10e-7,0.05)
plt.suptitle('BER vs SNR curve for Hamming encoder',fontsize=30)
ax[1].grid(True)
plt.show()
fig.savefig('hamming_awgn_fad')