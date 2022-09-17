# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:27:27 2021

@author: Dell
"""
import numpy as np
def hammingDist(str1, str2):
    i = 0
    count = 0
 
    while(i < len(str1)):
        if(str1[i] != str2[i]):
            count += 1
        i += 1
    return count

