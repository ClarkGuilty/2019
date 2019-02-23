#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:01:08 2019

@author: Javier Alejandro Acevedo Barroso
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#para el LiF
d = 2.014*1e-10 #metros  (110)

theta = np.arange(3.0,55,0.1)


#parte 4
datos = np.loadtxt('parte4(21022019).txt', skiprows = 5, delimiter ="	")
satan = datos[:,11]
satan[30] = (satan[29]+satan[31])/2

angulos4 = datos[:,0]
nombres = ["13kV","15kV","17kV","19kV","21kV","23kV","25kV","27kV","29kV", "31kV", "33kV", "35kV"]
voltajes = [datos[:,1],datos[:,2],datos[:,3],datos[:,4],datos[:,5],datos[:,6],datos[:,7],datos[:,8],datos[:,9],datos[:,10],datos[:,11],datos[:,12]]
dictionary = dict(zip(nombres, voltajes))



plt.figure(figsize=(10,10))
for nombre,voltaje in zip(nombres,voltajes):
#    voltaje = voltaje/voltaje.sum()
    plt.plot(angulos4,voltaje,label=nombre)

plt.xlim(3,6)
#plt.xlim(19,23)    
#plt.ylim(0,0.02)
plt.ylim(0,2000)
plt.legend()


#parte 3















