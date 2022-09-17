# -*- coding: utf-8 -*-
"""
Created on Sun May 23 00:33:55 2021

@author: Dell
"""
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 20:30:07 2021

@author: Dell
"""
import matplotlib.pyplot as plt
import numpy as np

dhd = [0.0107,0.00459025,0.001600375,0.0004345,8.05e-05,8.83333333333333e-06,1e-06,0]+ list(np.zeros(22))
snrdB = np.arange(0,7)
snr_lin = 10**(snrdB/10)
EbN0dBs1 = 10*np.log10(snr_lin*7/4);

snrdB = np.arange(0,8)
snr_lin = 10**(snrdB/10)
EbN0dBs2 = 10*np.log10(snr_lin*7/4)
EbN0dBs3 = 10*np.log10(snr_lin*15/11)
snrdB = np.arange(0,6)
snr_lin = 10**(snrdB/10)
EbN0dBs4 = 10*np.log10(snr_lin*21/11)

dhd = [0.0107,0.004767,0.0016575,0.0004575,8.95e-05,1.3e-05,1.5e-06]
shd = [0.0107,0.00459025,0.001600375,0.0004345,8.05e-05,8.83333333333333e-06,1e-06]
d_7_4 = [0.017287,0.00879,0.0036775,0.001228,0.0003055,5.6e-05,4.5e-06]#,0]
d_15_11 = [0.06871727, 0.04752864, 0.02938182, 0.01597636, 0.00768182,0.00304227, 0.001045  , 0.00030318]
d_21_11 = [2.160875e-03, 1.098750e-03, 3.196875e-04, 6.535000e-05,0.818750e-05, 0.5000000e-06]#,0.000000e+00 , 0.000000e+00]

fig, ax = plt.subplots(nrows=1,ncols = 1)
fig.set_figheight(10)
fig.set_figwidth(10)

ax.semilogy(EbN0dBs1,dhd,marker='o',linestyle='-',label='Hamming encoder-DL decoder')
ax.semilogy(EbN0dBs1,shd,marker='d',linestyle='-',label='Hamming encoder-soft decision decoder')
ax.semilogy(EbN0dBs1,d_7_4,marker='x',linestyle='-',label='(7,4) Autoencoder')
ax.semilogy(EbN0dBs3,d_15_11,marker='v',linestyle='-',label='(15,11) Autoencoder')
ax.semilogy(EbN0dBs4,d_21_11,marker='p',linestyle='-',label='(21,11) Autoencoder')

ax.legend(prop={"size":16})
ax.set_xlabel('$SNR(dB)$',fontsize=16)
ax.set_ylabel('BER',fontsize=16)
ax.set_title('BER vs SNR curves in AWGN channel',fontsize=20)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14)
#ax.set_xlim(2.2,10);ax.set_ylim(10e-7,0.05)
ax.grid(True)
plt.show()
fig.savefig('awgn_all')