# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 20:09:36 2021

@author: Dell
"""
# -*- coding: utorch-8 -*-
"""
Created on Thu Feb 25 14:28:10 2021

@author: Dell
"""
import torch,torchvision
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from model import Encoder,Decoder
from dataset import generate_dataset,make_datasets
from channel import noise
import os 
cwd = os.getcwd()
lamda = 1#e-3
encoder = Encoder()
decoder = Decoder()
n = 2**(11)
n = 16
 
def loss_object(C):
    size = C.size()
    n = size[0]
    D = torch.ones((n,n))
    C = torch.transpose(C,0,1)
    L = torch.ones((n,n))
    T = torch.matmul(torch.transpose(C,0,1),C)
    diag = torch.diagonal(T,0)
    L = L*diag
    L = L.T
    D = L-2*T+torch.transpose(L,0,1)
    diag = torch.eye(n,n)*1000
    D = D + diag
    return -1*lamda*torch.min(D)


def checkpoint_encoder():
    model_out_path = 'ckpts/model_encoder'
    path = os.path.join(cwd,model_out_path)
    if os.path.exists(path) != 1:
        os.mkdir(path)
    torch.save(encoder.state_dict(), path + '/weights')

def checkpoint_decoder():
    model_out_path = 'ckpts/model_decoder'
    path = os.path.join(cwd,model_out_path)
    if os.path.exists(path) != 1:
        os.mkdir(path)
    torch.save(decoder.state_dict(), path + '/weights')

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
encoder.apply(init_weights)
decoder.apply(init_weights)

train_x, train_y, test_x, test_y = generate_dataset(length = 1)
train_x = train_x[:n,0:5]
#train_x = make_datasets()
#train_x = 2*train_x - 1

train_x = torch.Tensor(train_x) # transform to torch tensor
dataset = TensorDataset(train_x) # create your datset
trainloader = DataLoader(dataset,batch_size=16,shuffle=True)

optimizer_encoder = optim.Adam(encoder.parameters(), lr=0.001)
optimizer_decoder = optim.Adam(encoder.parameters(), lr=0.001)

#optimizer = optim.SGD(encoder.parameters(), lr=0.001, momentum=0.9)
EPOCHS = 30000
minLossTrain = 999
minLossTrainEncoder = 999
criterion = torch.nn.MSELoss()
#encoder.load_state_dict(torch.load('ckpts/model_encoder_only/weights', map_location=lambda storage, loc: storage))

for i in range(EPOCHS):
    # only encoder
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs = data[0]
            msg = inputs[:,0:4]
            optimizer_encoder.zero_grad()
    
            outputs = encoder(msg)
            en_loss = loss_object(outputs)
    
            en_loss.backward()
            optimizer_encoder.step()
    
            running_loss += en_loss.item()
        # print('==> Only Encoder Epoch %d loss: %.7f' %
        #           (epoch + 1, running_loss))
        if minLossTrainEncoder > running_loss:
              minLossTrainEncoder = running_loss
              checkpoint_encoder()
    print(minLossTrainEncoder)
    ## joint training
    for epoch in range(100):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs = data[0]
            msg = inputs[:,0:4] 
            snr = inputs[:,4:5]
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
    
            encoded = encoder(msg)
            Es = torch.mean(encoded**2,axis = 1)
            #Es = torch.ones_like(Es)
            
            n = noise(snr,Es)
            r = n + encoded
            decoded = decoder(r)
            loss = criterion(msg,decoded)
    
            loss.backward()
            optimizer_decoder.step()
            optimizer_encoder.step()
    
            running_loss += loss.item()
        # print('Joint Model Epoch %d loss: %.7f' %
        #           (epoch + 1, running_loss))
        if minLossTrain > running_loss:
              minLossTrain = running_loss
              checkpoint_encoder()
              checkpoint_decoder()
              enco_loss = loss_object(encoded)
    print('===========================================')
    print('Encoder loss: %.7f || Decoder loss: %.7f ' % (enco_loss,minLossTrain))
    print('===========================================')