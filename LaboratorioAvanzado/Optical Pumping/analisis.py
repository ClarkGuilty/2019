#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 18:21:01 2019

@author: Javier Alejandro Acevedo Barroso
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


datos = np.loadtxt('datosZeemanDebil(07032019).txt', skiprows=1)
freq = datos[:,0]
p0 = datos[:,1]
p1 = datos[:,2]
p2 = datos[:,3]

m1, b1, r1, pp1, sigma1 = linregress(p1,freq)
m2, b2, r2, pp2, sigma2s = linregress(p2[:-1],freq[:-1])

Iprueba = np.linspace(0,1000,100)

plt.scatter(p2[:-1],freq[:-1])
plt.scatter(p1,freq)
plt.plot(Iprueba,m1*Iprueba+b1)
plt.plot(Iprueba,m2*Iprueba+b2)
plt.xlim(0,1000)
plt.ylim(0,200)