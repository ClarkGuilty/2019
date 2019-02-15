#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:01:08 2019

@author: Javier Alejandro Acevedo Barroso
"""


import numpy as np
import matplotlib.pyplot as plt


#para el LiF
d = 2.014*1e-10 #metros  (110)

theta = np.arange(3.0,55,0.1)


t1 = np.loadtxt('p1_LiF_T1(14022019).dat')
plt.scatter(theta[:len(t1)],t1)






















