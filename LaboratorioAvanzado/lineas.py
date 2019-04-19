#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:12:19 2019

@author: Javier Alejandro Acevedo Barroso
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz


def gauss(x,mean,std):
    return np.power(std**2*2*np.pi,-1/2)*np.exp(-0.5*((x-mean)/std)**2)

line = 4481
sigma0 = 15
sigma1 = 20
sigma2 = 30
x = np.linspace(line-60, line+60,121)
y0 = gauss(x,line,sigma0)
y1 = gauss(x,line,sigma1)
y2 = gauss(x,line,sigma2)

I0 = trapz(y0,x)
I1 = trapz(y1,x)
I2 = trapz(y2,x)
x 
plt.figure(figsize=(10,7))
plt.plot(x/10,y0/y0[x==line], label = "Perfil sin rotación",linewidth=5)
plt.plot(x/10,y1/y0[x==line], label = "Perfil con baja rotación",linewidth=5)
plt.plot(x/10,y2/y0[x==line], label = "Perfil con alta rotación",linewidth=5)
plt.title(r"Mg 4481 a diferentes $v_{rot}$", fontsize=20)
plt.xlabel(r'Longitud de onda [nm]', fontsize=20)
plt.ylabel("Intensidad relativa", fontsize=20)
plt.legend()
plt.savefig("Mg4481 ref.png",dpi = 1000, transparent=True)


