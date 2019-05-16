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
factor = 150
#vReal[4] -= factor
#vMed[4] -= factor
#sigmaReal[4] /= factor
#sigmaMed[4] /= factor
espectral = ['A1V', 'B8I', 'B1II', 'B2III','A1V' ]
f,(ax,ax2) = plt.subplots(2,1,sharex=True, facecolor='w')
for i, color in zip(xpos[:-1],["blue","indigo","forestgreen","teal"]):
            
      ax2.errorbar([xpos[i]],vMed[i], yerr=[sigmaMed[i]],color = 'k', fmt='o')
#      plt.scatter([xpos[i]], vMed[i],color = color, label = estrellas[i] + " medido")
#      plt.scatter([xpos[i]-0.05], vReal[i], marker= '^',color = 'r', s = 50)
      ax2.errorbar([xpos[i]-0.065], vReal[i], color = 'r',yerr=[sigmaReal[i]], fmt='^')
#ax.errorbar([xpos[4]],vMed[4], yerr=[sigmaMed[4]],color = 'k', fmt='o')
#ax.errorbar([xpos[4]-0.05], vReal[4], color = 'r',yerr=[sigmaReal[4]], fmt='^')
ax.yaxis.set_tick_params(labelsize=12)
ax2.yaxis.set_tick_params(labelsize=12)
ax.scatter([xpos[4]-0.065], vReal[4], marker= '^',color = 'r', s = 50)
ax.errorbar([xpos[4]-0.065],vReal[4], yerr=[sigmaReal[4]],color = 'r', fmt='^', label = 'Valor de la literatura')
ax.errorbar([xpos[4]],vMed[4], yerr=[sigmaMed[4]],color = 'k', fmt='o', label = 'Valor medido')
plt.xticks(xpos,estrellas+'\n'+espectral,fontsize=14)
#ax.yticks(fontsize=14)
#ax2.yticks(fontsize=14)
plt.ylabel(r"                   $V \sin{i}$ [km/s]", fontsize=18,labelpad=10)

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop='off', labelright = True)  # don't put tick labels at the top
ax2.tick_params(labeltop='off', labelright = True)  # don't put tick labels at the top
ax.xaxis.set_ticks_position('none') 
ax2.yaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
#ax2.yaxis.tick_right()
#ax.yaxis.tick_right()
#plt.axhline(80,linestyle='--', color = 'black')
#plt.yticks((1+np.arange(7))*20,[20,40,60,80,100+factor,120+factor,140+factor],fontsize=12)
ax.legend(prop={'size': 14})
#ax.title.set_text("Velocidades de rotación")
f.suptitle("Resultados comparados con la literatura", fontsize=19)
f.subplots_adjust(hspace=0.1) 
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonals

#ax2.plot((0,1 ), (1.05, 1.05 ), **kwargs)  # bottom-right diagonals

f.savefig("resultados.png",dpi = 1000)



