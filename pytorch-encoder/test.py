# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 23:12:38 2021

@author: Dell
"""
import torch,torchvision
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from model import Encoder
from dataset import generate_dataset
from channel import noise
import os 
np.set_printoptions(precision=1)
model_encoder = Encoder()

model_encoder.load_state_dict(torch.load('ckpts/model_encoder_only/weights', map_location=lambda storage, loc: storage))
snr = 10**(11/10)
msg = []
n = 4
b = 7
k = 2**n
for i in range(k):
    m = [int(x) for x in format(i, 'b').zfill(n)]
    msg.append(m)
    
testset_x = np.zeros((k,n))
for j in range(k):
    testset_x[j][0:n] = msg[j]
    #testset_x[j][4:5] = snr
testset_x = 2*testset_x-1
testset_x =testset_x.reshape((k,n))
testset_x = torch.Tensor(testset_x) # transform to torch tensor

en1 = model_encoder(testset_x)

enc = en1.detach().numpy() >= 0
enc = 1*enc
#enc = enc.astype(np.float32)
#enc = en[0]
for i in range(k):
  print(msg[i],"==>",enc[i])
enc = torch.Tensor(en1)
enc = enc >0
enc = 2*enc - 1
D = torch.ones((k,k))
C = torch.transpose(enc,0,1)
L = torch.ones((k,k))
T = torch.matmul(torch.transpose(C,0,1),C)
diag = torch.diagonal(T,0)
L = L*diag
L = L.T
D = L-2*T+torch.transpose(L,0,1)
diag = torch.eye(k,k)*100
D = D + diag
print("Minimum hamming distance is "+str(torch.min(D)))