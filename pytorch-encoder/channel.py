# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 13:06:55 2021

@author: Dell
"""
import numpy as np
import torch
def noise(snr,Es):
    #noise_power = torch.reshape(Es/torch.reshape(snr,(-1,)),(-1,1))
    noise_power = Es/snr
    sigma = torch.sqrt(noise_power)
    # std_dev = torch.ones_like(noise_power)
    # mean = torch.zeros((1,7))
    n = torch.randn(1000000, 7)
    n = sigma * n 
    return n
