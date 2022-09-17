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
    diag = torch.eye(n,n)*12
    D = D + diag
    return -1*lamda*torch.min(D)



def checkpoint_encoder():
    model_out_path = 'ckpts/model_encoder'
    path = os.path.join(cwd,model_out_path)
    if os.path.exists(path) != 1:
        os.mkdir(path)
    torch.save(encoder.state_dict(), path + '/weights')

def checkpoint_decoder():
    model_out_path = 'ckpts/model_decoder_DL'
    path = os.path.join(cwd,model_out_path)
    if os.path.exists(path) != 1:
        os.mkdir(path)
    torch.save(decoder.state_dict(), path + '/weights')

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
        
def bool2int(x):
    p = []
    for i in x :
        c = int(i[0])*8+int(i[1])*4+int(i[2])*2+int(i[3])*1
        p.append(c)
    return np.array(p)
def convertToOneHot(vector, num_classes=None):
    result = torch.zeros(vector.shape[0],num_classes)
    for i in range(vector.shape[0]):
        result[i, vector[i]] = 1
    return result#.type(torch.)

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


train_x, train_y, test_x, test_y = generate_dataset(length = 1)
train_x = train_x[:,0:5]
train_x[:,0:4] = 2*train_x[:,0:4] - 1
# inputSyms = np.random.randint(low=0, high = 2, size=10**6) 
# inputSyms = 2*inputSyms - 1
# inputSyms = inputSyms.reshape(-1,4)
train_x = torch.Tensor(train_x) # transform to torch tensor
dataset = TensorDataset(train_x) # create your datset
trainloader = DataLoader(dataset,batch_size=16,shuffle=False)

optimizer_decoder = optim.Adam(decoder.parameters(), lr=0.001)

EPOCHS = 100000
minLossTrain = 999
criterion = torch.nn.MSELoss()
bce = torch.nn.BCELoss()

#encoder.load_state_dict(torch.load('ckpts/model_encoder_only/weights', map_location=lambda storage, loc: storage))
decoder.load_state_dict(torch.load('ckpts/model_decoder_DL/weights', map_location=lambda storage, loc: storage))

snrdB = np.arange(0,8)
snr_lin = 10**(snrdB/10)
for epoch in range(EPOCHS):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs = data[0]
        msg = inputs[:,0:4] 
        msg1 = (msg+1)/2
        msg1 = bool2int(msg1)
        size = msg.size()
        tmp=np.random.randint(8, size=1)
        snr = snr_lin[tmp]#inputs[:,4:5]
        #optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        encoded = torch.Tensor(h[msg1])
        #encoded = encoder(msg)

        Es = torch.mean(torch.mean(encoded**2,axis = 1))

        noise_power = Es/snr
        sigma = torch.sqrt(noise_power)
        n = sigma*(torch.randn(16, 7))
        r = n + encoded
        r = r.float()
        decoded = decoder(r)
        loss = criterion(msg,decoded)
        #loss = bce(decoded,convertToOneHot(msg1,16))

        loss.backward()
        optimizer_decoder.step()
        #optimizer_encoder.step()

        running_loss += loss.item()
    print('Joint Model Epoch %d loss: %.7f' %
              (epoch + 1, running_loss))
    if minLossTrain > running_loss:
          minLossTrain = running_loss
          #checkpoint_encoder()
          checkpoint_decoder()
          enco_loss = loss_object(encoded)
    # print('===========================================')
          print('Encoder loss: %.7f || Decoder loss: %.7f ' % (enco_loss,minLossTrain))
    # print('===========================================')
    