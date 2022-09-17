# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:51:28 2021

@author: Dell
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 7)
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(7)
        self.lr = nn.LeakyReLU()
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = F.relu(self.fc4(x))
        x = self.bn4(x)
        x = self.fc5(x)

        #x = torch.tanh(self.fc5(x))
        x = self.bn5(x)
        x = torch.tanh(x)
        m = nn.LayerNorm(x.size()[1:])
        x = m(x)
        return x

class Encoder11(nn.Module):
    def __init__(self):
        super(Encoder11, self).__init__()
        self.fc0 = nn.Linear(11, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 21)
        self.dropout = nn.Dropout(p=0.2)
        self.bn0 = nn.BatchNorm1d(512)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(32)
        self.bn5 = nn.BatchNorm1d(21)
        self.lr = nn.LeakyReLU()
    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = self.bn0(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = F.relu(self.fc4(x))
        x = self.bn4(x)
        x = self.fc5(x)

        #x = torch.tanh(self.fc5(x))
        x = self.bn5(x)
        x = torch.tanh(x)
        m = nn.LayerNorm(x.size()[1:])
        x = m(x)
        return x
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.fc1 = nn.Linear(4, 32)
#         self.fc2 = nn.Linear(32, 16)
#         self.fc3 = nn.Linear(16, 7)


#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))

#         x = torch.tanh(self.fc3(x))
#         return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(7, 512)
        self.fc10 = nn.Linear(512, 128)
        self.fc11 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn10 = nn.BatchNorm1d(128)
        self.bn11 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(8)
        self.lr = nn.LeakyReLU()
        self.sm = nn.Softmax()
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = torch.relu(self.fc10(x))
        x = self.bn10(x)
        x = torch.relu(self.fc11(x))
        x = self.bn11(x)
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = torch.relu(self.fc3(x))
        #x = self.bn3(x)
        x = torch.relu(self.fc4(x))
        #x = self.bn4(x)
        x = torch.tanh(self.fc5(x))

        return x
