# -*- coding: utorch-8 -*-
"""
Created on Thu Feb 25 14:28:10 2021

@author: Dell
"""
import torch,torchvision
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable

from model import Encoder,Encoder11
from dataset import generate_dataset,make_datasets
from channel import noise
import os 
import matplotlib.pyplot as plt
cwd = os.getcwd()
lamda = 1
encoder = Encoder11()
encoder = encoder#.cuda(0)

 
def loss_object(C):
    size = C.size()
    #C = Variable(C, requires_grad=True).cuda(0)

    n = size[0]
    #D = torch.ones((n,n), requires_grad=True)
    #C = torch.transpose(C,0,1)
    L = torch.ones((n,n))#.cuda(0)
    T = torch.matmul(C,torch.transpose(C,0,1))
    #L = Variable(L, requires_grad=True).cuda(0)
    #T = Variable(T, requires_grad=True).cuda(0)
    diag = torch.diagonal(T,0)
    L = L*diag
    D = L-2*T+L.T
    D =  D[D>0]
    #D = Variable(D, requires_grad=True).cuda(0)
    # norm_D = torch.norm(D)
    # norm_D_nuc = torch.norm(D,'nuc')
    #D = torch.tril(D,-1)
    #D = D + (torch.eye(2**n)*100)#.cuda(0)
    min_D = torch.min(D)
    x = D[D == min_D]
    #x = Variable(x, requires_grad=True).cuda(0)
    #return -1*lamda*norm_D
    #print(s[0])
    ham = -1*torch.mean(x)
    #ham = Variable(ham, requires_grad=True).cuda(0)
    return ham

def euc_dist(C):
    D = torch.cdist(C, C, p=2, compute_mode='use_mm_for_euclid_dist')
    D = D + (torch.eye(16)*100)#.cuda(0)
    
    #D = torch.tril(D,-1)
    
    k = torch.min(D)
    return -1*k
def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    print(ave_grads)
    # plt.plot(ave_grads, alpha=0.3, color="b")
    # plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    # plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(xmin=0, xmax=len(ave_grads))
    # plt.xlabel("Layers")
    # plt.ylabel("average gradient")
    # plt.title("Gradient flow")
    # plt.grid(True)
    
def loss_loop(C):
    min_d = 100
    for i in range(16):
        for j in range(16):
            if i != j:
                D  = ((C[i]-C[j])**2).sum(axis=0)
                if min_d > D:
                    min_d = D
    return -1*lamda*min_d

def checkpoint():
    model_out_path = 'ckpts/model_encoder(21,11)_new'
    path = os.path.join(cwd,model_out_path)
    if os.path.exists(path) != 1:
        os.mkdir(path)
    torch.save(encoder.state_dict(), path + '/weights')


msg = []
n = 11
b = 21
k = 2**n
for i in range(k):
    m = [int(x) for x in format(i, 'b').zfill(n)]
    msg.append(m)
msg = np.array(msg)
#train_x, train_y, test_x, test_y = generate_dataset(length = 1)
train_x = msg[:,0:n]
train_x = 2*train_x - 1

train_x = torch.Tensor(train_x) 
dataset = TensorDataset(train_x) 
trainloader = DataLoader(dataset,batch_size=2**n,shuffle=False)

optimizer = optim.Adam(encoder.parameters(), lr=0.001)
EPOCHS = 500000
minLossTrainEncoder = 999

encoder.load_state_dict(torch.load('ckpts/model_encoder(21,11)_new
                                   
                                   /weights', map_location=lambda storage, loc: storage))

for epoch in range(EPOCHS):  # loop over the dataset multiple times
    running_loss = 0.0
    encoder.train()
    for i, data in enumerate(trainloader, 0):

        inputs = data[0]
        inputs = Variable(inputs)#.cuda(0)
        optimizer.zero_grad()

        outputs = encoder(inputs)
        #inputs = Variable(outputs).cuda(0)
        loss = loss_object(outputs)
        #loss = euc_dist(outputs)
        loss.backward()
        optimizer.step()
        #plot_grad_flow(encoder.named_parameters())
        # if epoch == 1000:
        #     print(encoder.fc3.weight.grad)
        #     print(encoder.fc3.bias.grad)
        #     print(encoder.fc2.weight.grad)
        #     print(encoder.fc2.bias.grad)
        #     f
        running_loss += loss.item()
    print('Epoch %d loss: %.7f'%
              (epoch + 1, running_loss))
    if minLossTrainEncoder > running_loss:
          minLossTrainEncoder = running_loss
          checkpoint()

