#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 21:02:40 2019

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt

#Returns flux in F0 units
def flux(m):
      return np.power(10,-2*m/5)

#Takes flux in F0 and returns the magnitude
def magnitude(f):
      return -2.5*np.log10(f)

def sumOfMagnitudes(emes):
      totalF = 0
      for m in emes:
           totalF += flux(m)
      return magnitude(totalF)

def F3(m1,m2):
      f1 = flux(m1)
      f2 = flux(m2)
      return f1**2+f2**2+2*f1*f2-f1-f2 

def M3(m1,m2):
      return magnitude(F3(m1,m2))

def R(mu,A=0):
      return 10**((mu+5-A)/5)

m1 = -1
m2 = 2
m3 = M3(m1,m2)



########################

def gaussian(x,mean,sigma):
      return np.power(2*np.pi*sigma**2,-1/2)*np.exp(-0.5*((x-mean)/sigma)**2)


labels = ['J', 'H', r'K', 'L']
centers = np.array([1220,1630,2190,3450])
FWHM = np.array([213,307,390,472])

sigma = FWHM/(2*np.sqrt(2*np.log(2)))

x = np.linspace(800,4500, 600)
linestyles = ['--','-.','dotted','-']

for label,mean,s,linestyle in zip(labels,centers,sigma,linestyles):
      #plt.plot(3e8/x/1e6, gaussian(x,mean,s), label=label)
      plt.plot(1e7/x, gaussian(x,mean,s), label=label, linestyle=linestyle)

plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
#    top=False,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off

plt.title('Respuesta de los filtros JHKS', size=20)
plt.ylabel("Respuesta", size = 18)
plt.xlabel(r"$\lambda$ [nm]", size= 18)
plt.legend()













