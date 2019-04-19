#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 23:21:52 2019

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import factorial
from numpy import random

def stdf(arr):
      return np.sqrt(np.sum((arr-np.mean(arr))**2)/(arr.size-1))

def gauss(x,mean,std):
      return np.sqrt(1./(2*std*std*np.pi))*np.exp(-(x-mean)**2/(2*std**2))

def err(xi,xf,mean,std):
      return quad(gauss,xi,xf,args=(mean,std))[0]

def poisson(N,meanN):
      return np.exp(-meanN)*meanN**N/factorial(N)

print("punto 3.4")
mean = 10
std = 1
m = 4
print("La fracción de datos por fuera de {:d} es {:.8f}".format(m,err(mean-m*std,mean+m*std,mean,std)))
m=5
print("La fracción de datos por fuera de {:d} es {:.8f}".format(m,err(mean-m*std,mean+m*std,mean,std)))
m=0.674489
print("La fracción de datos dentro de {:.2f} sigma es {:.8f}".format(m,err(mean-m*std,mean+m*std,mean,std)))
m=3.291
print("La fracción de datos dentro de {:.2f} es {:.8f}".format(m,err(mean-m*std,mean+m*std,mean,std)))



print("punto 3.5")
mean = 502
std = 14
print("La probabilidad de que un paquete contenga menos de 500 es: {:.3f}".format(err(0,500,502,14)))

print("La probabilidad de que un paquete contenga más de 530 es: {:.3f}".format(err(530,np.inf,502,14)))
print("De 1000, se espera que {:d} pesen más o igual a 530".format(int(err(530,np.inf,502,14)*1000)))


print("punto 3.6")
data = np.array([45.7, 53.2, 48.4, 45.1, 51.4, 62.1, 49.3])
#mean = np.mean(data)
#std = stdf(data)

def cChauvenet(n,arr):
      mean = np.mean(arr)
      stdev = stdf(arr)
      x = arr[n]
      diff = np.abs(x-mean)
      print(x, mean -diff, mean+diff)
      pout = 1-err(mean-diff,mean+diff,mean,stdev)
#      print(pout)
      return pout*arr.size
      

xtest = np.linspace(np.min(data),np.max(data),100)
print("n_out es", cChauvenet(5,data))
print("al ser menor que 0.5, se descarta el dato.")

plt.figure()
plt.hist(data, density=True)
plt.plot(xtest,gauss(xtest,np.mean(data),stdf(data)))



print("punto 3.8")
total = 270
rate = 270/60


print("La media de conteos por segundo es: {:.3f}".format(rate))
print("El error es {:.3f}".format(np.sqrt(rate)))
print("El error fraccional es {:.3f}".format(np.sqrt(1./rate)))

time = 15*60 #s
xtest = np.arange(0,14)
#y = poisson(xtest,rate)*60
y = random.poisson(rate,time)*time



print(np.mean(y))

plt.figure()
plt.hist(y/time,bins = 15,density=True)
plt.plot(xtest,poisson(xtest,rate))

print("El conteo esperado en 15 minutos es: {:d}".format(int(rate*time)))
print("La probabilidad de obtener 270 x 15 conteos es {:.3f}".format(poisson(rate,rate)))




print("punto 3.9")
plt.figure()
xtest = np.arange(0,70)
ymax = 0
for n in [5,15,30,80,200]:
      y = random.poisson(35,n)
      if(np.max(y) > ymax):
            ymax = np.max(y)
      plt.hist(y, density = True, label = n)

xgauss = np.linspace(0,ymax,100)
plt.plot(xgauss, gauss(xgauss,35,np.sqrt(35)),color = 'lightgrey')

plt.legend()




















































