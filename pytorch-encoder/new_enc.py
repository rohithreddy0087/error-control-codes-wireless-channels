# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 22:59:52 2021

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

from model import Encoder
from dataset import generate_dataset,make_datasets
from channel import noise
import os 
cwd = os.getcwd()
lamda = 1e-3
encoder = Encoder()

#n = 2**(11)
n = 16
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




def checkpoint():
    model_out_path = 'ckpts/model_encoder_only'
    path = os.path.join(cwd,model_out_path)
    if os.path.exists(path) != 1:
        os.mkdir(path)
    torch.save(encoder.state_dict(), path + '/weights')

#h = 2*h-1

train_x, train_y, test_x, test_y = generate_dataset(length = 1)
train_x = train_x[:n,0:4]

train_x = torch.Tensor(train_x) # transform to torch tensor
train_y =  torch.Tensor(h)

dataset = TensorDataset(train_x,train_y) # create your datset
trainloader = DataLoader(dataset,batch_size=16,shuffle=False)

optimizer = optim.Adam(encoder.parameters(), lr=0.001)
#optimizer = optim.SGD(encoder.parameters(), lr=0.001, momentum=0.9)

mse = torch.nn.MSELoss()
EPOCHS = 30000
minLossTrainEncoder = 999

#encoder.load_state_dict(torch.load('ckpts/model_encoder_only/weights', map_location=lambda storage, loc: storage))

for epoch in range(EPOCHS):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        inputs = data[0]
        labels = data[1]
        optimizer.zero_grad()

        outputs = encoder(inputs)
        loss = mse(outputs,labels)
        
        loss.backward()
        optimizer.step()
        
        # if epoch == 10:
        #     print(encoder.fc3.weight.grad)
        #     print(encoder.fc3.bias.grad)
        #     print(encoder.fc4.weight.grad)
        #     print(encoder.fc4.bias.grad[0])
        #     f
        running_loss += loss.item()
    print('Epoch %d loss: %.7f'%
              (epoch + 1, running_loss))
    if minLossTrainEncoder > running_loss:
          minLossTrainEncoder = running_loss
          checkpoint()

