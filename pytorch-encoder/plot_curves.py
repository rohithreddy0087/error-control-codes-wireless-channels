# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 22:57:56 2021

@author: Dell
"""

import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from scipy.special import erfc #erfc/Q function
import pandas as pd  

snrdB = np.arange(0,8)
snr_lin = 10**(snrdB/10)
EbN0dBs = 10*np.log10(snr_lin*7/4);
BER_theory = 0.5*erfc(np.sqrt(10**(EbN0dBs/10)))

BER_soft_th = [0.27029141691578,0.14402420297513,0.067112994075819,0.0265243961872963,0.00855968658837643,
            0.00215044674848217,0.000395695638620641,4.92663626543647e-05,3.74397276166584e-06,1.52105077059930e-07,
            2.79355977572752e-09]
df11 = pd.read_csv('hamming_ber.csv')  
ber = df11['BER'] 

df = pd.read_csv('model_sim_hamming.csv')  
BER_sim = df['BER']

df1 = pd.read_csv('model_sim_DL.csv')  
BER_sim_AE = df1['BER']

#fig = plt.figure(figsize=(10,8))
fig,ax = plt.subplots(nrows=1,ncols = 1)
fig.set_figheight(15)
fig.set_figwidth(15)
ax.semilogy(EbN0dBs,BER_sim,color='r',marker='o',linestyle='-',label='Trained decoder only ham')
ax.semilogy(EbN0dBs,BER_theory,marker='d',linestyle='-',label='BPSK Theory')
#ax.semilogy(EbN0dBs,ber[:,0],marker='s',linestyle='-',label='Hamming matlab hard')
ax.semilogy(EbN0dBs,ber,marker='h',linestyle='-',label='Hamming soft')
#ax.semilogy(EbN0dBs,BER_soft_th,marker='v',linestyle='-',label='Hamming soft theory')
ax.semilogy(EbN0dBs,BER_sim_AE,marker='x',linestyle='--',label='(7,4) Autoencoder')

ax.set_xlabel('$E_b/N_0(dB)$');ax.set_ylabel('BER ($P_b$)')
ax.set_title('Probability of Bit Error using FCNN Auto-encoder over AWGN channel')
ax.set_xlim(1,12);
ax.set_ylim(10e-7,0.2);
ax.grid(True);
ax.legend();plt.show()
fig.savefig('saved_plot')

###################################################
dhd = [0.08252585, 0.06854725, 0.0569364 , 0.04729225, 0.038168,
       0.03262797, 0.02710129, 0.02251075, 0.01869778, 0.01553066,
       0.01290001, 0.011271495, 0.0099    , 0.00739248, 0.00644031,
       0.00530023, 0.00463633, 0.00351876, 0.00292274, 0.00252767,
       0.00201646, 0.001749 , 0.0013912 , 0.001166555, 0.00098982,
       0.00079724, 0.0006822 , 0.00057003, 0.00045687, 0.00037948]
shd = [0.056967  , 0.04735   , 0.039714  , 0.032279  , 0.0265725 ,
       0.0218405 , 0.0171645 , 0.0139705 , 0.011615  , 0.0091185 ,
       0.007389  , 0.005867  , 0.004713  , 0.0036915 , 0.0030015 ,
       0.0023985 , 0.00186   , 0.0014685 , 0.001189  , 0.000982  ,
       0.0007865 , 0.0006325 , 0.000473  , 0.0003915 , 0.0003    ,
       0.0002205 , 0.0001995 , 0.00015325, 0.0001265 , 0.00010025]

d_7_4 = [0.11343497, 0.09100414, 0.0819017 , 0.06455754, 0.05349916,
       0.04733503, 0.03674066, 0.03044717, 0.02523173, 0.02090966,
       0.01732794, 0.0155975, 0.0119    , 0.00996159, 0.00847235,
       0.00677247, 0.00591238, 0.00465101, 0.00375431, 0.00329409,
       0.00264696, 0.00229355, 0.0018178 , 0.00160642, 0.00129838,
       0.00103454, 0.000915733, 0.00071047, 0.00058877, 0.00048792]

d_15_11 = [0.19614927, 0.16243765, 0.13451995, 0.1220039, 0.09225432,
       0.07839883, 0.06326838, 0.05439462, 0.04338971, 0.03893244,
       0.02975683, 0.02464261, 0.02140735, 0.01754    , 0.01429544,
       0.01159009, 0.0099813, 0.00794853, 0.00658244, 0.00585113,
       0.00451426, 0.00393841, 0.0032959 , 0.00256382, 0.00212318,
       0.0017827, 0.00140608, 0.00120583, 0.0010859, 0.00082696]

d_21_11 = [3.17643527e-02, 2.51004075e-02, 2.07940943e-02, 1.78244164e-02,
       1.36125663e-02, 1.15138716e-02, 8.94127834e-03, 7.29007876e-03,
       5.83364516e-03, 4.79997838e-03, 3.86891515e-03, 3.08986859e-03,
       2.590000000e-03, 2.09273975e-03, 1.63659044e-03, 1.39415861e-03,
       1.07137130e-03, 8.96842127e-04, 7.11358410e-04, 5.87466213e-04,
       4.64134586e-04, 3.75483911e-04, 3.00566109e-04, 2.49186806e-04,
       1.96761448e-04, 1.79198880e-04, 1.28807161e-04, 1.09217346e-04,
       8.83218272e-05, 6.82244446e-05]

d = [0.0107,0.00459025,0.001600375,0.0004345,8.05e-05,8.83333333333333e-06,1e-06,0] + list(np.zeros(22))

##############################################################

hd = [0.0107,0.004767,0.0016575,0.0004575,8.95e-05,1.3e-05,1.5e-06,0]
d = [0.0107,0.00459025,0.001600375,0.0004345,8.05e-05,8.83333333333333e-06,1e-06,0]
ber_7_4 = [0.017287,0.00879,0.0036775,0.001228,0.0003055,5.6e-05,4.5e-06,0]
ber_15_11 = [0.06871727, 0.04752864, 0.02938182, 0.01597636, 0.00768182,0.00304227, 0.001045  , 0.00030318]
ber_21_11 = [4.18831818e-02, 2.37136364e-02, 1.15018182e-02, 4.76909091e-03,1.61363636e-03, 4.63636364e-04, 1.12272727e-04, 1.81818182e-05]
ber_21_11 = [2.160875e-03, 1.098750e-03, 3.196875e-04, 6.535000e-05,0.818750e-05, 0.5000000e-06,0.000000e+00 , 0.000000e+00]


################################################################

dhd = [0.10650365, 0.08497606, 0.07267156, 0.06002947, 0.05258662,
       0.04196043, 0.03383487, 0.02894889, 0.02308684, 0.02107061,
       0.01575304, 0.01371261, 0.01074891, 0.00897901, 0.0078344 ,
       0.00605849, 0.00520454, 0.00413394, 0.00341479, 0.00302075,
       0.00253005, 0.00192471, 0.00155988, 0.0013133 , 0.00104484,
       0.00091612, 0.00074023, 0.00065146, 0.00050509, 0.00044722]

d_7_4 = [0.14377993, 0.11576768, 0.09810661, 0.08703978, 0.06994194,
       0.05529658, 0.04867707, 0.037731  , 0.03196723, 0.02574532,
       0.021726661, 0.01756703, 0.01451103, 0.01298666, 0.00990144,
       0.00887896, 0.00655613, 0.00578082, 0.00460997, 0.00400801,
       0.00304556, 0.00279835, 0.00219634, 0.00177296, 0.001536453,
       0.00122076, 0.00099931, 0.00087546, 0.00074186, 0.00056325]
d_15_11 = [0.26200903, 0.21942941, 0.1789789 , 0.14767816, 0.12798776,
       0.10076652, 0.08723697, 0.0689569 , 0.05679581, 0.04891549,
       0.03895398, 0.03201225, 0.02644334, 0.0228432 , 0.01804331,
       0.01490446, 0.01281165, 0.01016989, 0.00845072, 0.00693931,
       0.00593213, 0.00479496, 0.00391126, 0.00343085, 0.0026688 ,
       0.00228453, 0.00182103, 0.00160424, 0.00124256, 0.0010264 ]

d_21_11 = [0.05082296, 0.04312065, 0.03367055, 0.02721907, 0.02198011,
       0.01762219, 0.01495805, 0.01113613, 0.00903383, 0.00785197,
       0.00631026, 0.00454379, 0.004     , 0.00328638, 0.00231854,
       0.00201865, 0.00175419, 0.00139695, 0.00112217, 0.00090995,
       0.00075462, 0.00057437, 0.00048091, 0.0003691 , 0.00030482,
       0.00022472, 0.00020109, 0.00016075, 0.00013491, 0.00010116]
shd = [0.07378   , 0.0621875 , 0.051535  , 0.042611  , 0.0349785 ,
       0.028116  , 0.0231475 , 0.018802  , 0.0152185 , 0.0120985 ,
       0.0095705 , 0.007715  , 0.0062255 , 0.005078  , 0.0040105 ,
       0.0031135 , 0.002475  , 0.002035  , 0.0015925 , 0.001297  ,
       0.000928  , 0.0007775 , 0.00066   , 0.000467  , 0.0004085 ,
       0.0002985 , 0.0002365 , 0.000216  , 0.00018   , 0.00013475]