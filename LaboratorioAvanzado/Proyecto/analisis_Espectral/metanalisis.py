#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 18:02:33 2019

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

raw = pd.read_table('resultados.txt')
llaves =  raw.keys()
tabla = raw.get_values()
estrellas  = tabla[:,0]
tabla = np.delete(tabla,0,1)
print(tabla)

vReal = np.array(tabla[:,2])
vMed = np.array(tabla[:,0])
sigmaReal = tabla[:,3]
sigmaMed = tabla[:,1]
#print(vReal)

plt.errorbar(vMed,vReal,xerr=sigmaMed,yerr=sigmaReal,fmt='o')
plt.title("Comparación con la literatura", fontsize = 18)
plt.xlabel(r"Velocidad de rotación de la literatura", fontsize=15)
plt.ylabel(r"Velocidad de rotación de medida",fontsize=15)
x = np.linspace(0,263)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(x,x, label = 'y = x')
plt.savefig("escalaComparacion.png", dpi = 1000)
from scipy.stats import linregress

m,b,r,p,sigma = linregress(vMed.astype(np.double),vReal.astype(np.double))


plt.plot(x,m*x+b)

plt.figure()
xpos = np.arange(len(vReal))
vReal[4] /= 2
vMed[4] /= 2
for i, color in zip(xpos,["blue","indigo","forestgreen","teal", "k"]):
      
      plt.errorbar([xpos[i]],vMed[i], yerr=[sigmaMed[i]],color = 'k', fmt='o')
#      plt.scatter([xpos[i]], vMed[i],color = color, label = estrellas[i] + " medido")
#      plt.scatter([xpos[i]-0.05], vReal[i], marker= '^',color = 'r', s = 50)
      plt.errorbar([xpos[i]-0.05], vReal[i], color = 'r',yerr=[sigmaReal[i]], fmt='^')
plt.scatter([xpos[i]-0.05], vReal[i], marker= '^',color = 'r', s = 50, label = 'Valor de la literatura')
plt.errorbar([xpos[i]],vMed[i], yerr=[sigmaMed[i]],color = color, fmt='o', label = 'Valor medido')
plt.xticks(xpos,estrellas,fontsize=13)
plt.ylabel(r"Velocidad de rotación", fontsize=15)
plt.axhline(80,linestyle='--', color = 'black')
plt.yticks((1+np.arange(7))*20,[20,40,60,80,100*2,120*2,140*2],fontsize=12)
plt.legend()
plt.savefig("resultados.png",dpi = 1000)
#plt.yticks()


