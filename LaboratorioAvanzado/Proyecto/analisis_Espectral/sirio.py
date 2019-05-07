#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:10:21 2019

@author: Javier Alejandro Acevedo Barroso
"""


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u

espectre = np.loadtxt("SirioProm.dat")
star = 'Sirio'

lamb = espectre[:,0]
y4 = espectre[:,1]

dpi = 300
#plt.plot(lamb,y4)
#conv0 = 2/(y1[202]+y1[203])
element = 'H'
line = 4861 #
lineS = str(line)
ii = np.where(np.logical_and(lamb<line+7,lamb>line-7))

#plt.plot(lamb,y4/conv0*1e-7)
plt.title(element+" "+lineS+" en "+star+" transformada" ,fontsize=20)
#plt.ylabel("Flujo [J cm$^{-2}$ $\AA^{-1}$ s$^{-1} \ 10^{16}]$", fontsize=18)
#plt.xlabel(r'Longitud de onda [$\AA$]', fontsize=18)
linea = y4[ii]
lambLinea = lamb[ii]
lambLinea = (lamb[ii] - line)/line
#plt.plot(lamb[ii],y4[ii]/conv0)
#plt.xlim(min(lamb),6700)
linea = np.max(linea) - linea
linea = linea/np.max(linea)
plt.plot(lambLinea,linea)
plt.ylabel("Intensidad relativa [$F/F_{\lambda = "+lineS+"}$]", fontsize=18)
plt.xlabel(r'($\lambda - \lambda_m)/\lambda_m$', fontsize=18)
#plt.xlabel(r'$\frac{\lambda - \lambda_m}{\lambda_m}$', fontsize=18)
#plt.savefig("HR1544Mg4481B.png", dpi = dpi, bbox_inches='tight')

from numpy import fft
from scipy.integrate import trapz


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx



#linea = linea/linea[lambLinea==0]
#linea = np.max(linea) - linea
linea = linea/np.max(linea)
deltaSigma = 1.0/(linea.size * (lamb[1]-lamb[0]))
lineas = fft.fft(linea)
#freq = fft.fftfreq(linea.size,deltaSigma)
freq = fft.fftfreq(linea.size, d = lambLinea[1]-lambLinea[0])
iimax = 100
#plt.xlim(0.008,0.00825)
plt.figure()
plt.xlabel("Frecuencias [hz]", fontsize=18)
plt.ylabel("FFT normalizada", fontsize=18)
plt.title("FFT de H 4861 en "+star, fontsize=18)
plt.plot(freq,lineas.real/np.max(lineas.real),color = 'r')
plt.scatter(freq,lineas.real/np.max(lineas.real), color = 'orange')
#plt.xlim(800,850)
#plt.ylim(-0.1,0.1)
plt.savefig(star+element+lineS+".png", dpi = dpi, transparent=True,bbox_inches='tight')
#plt.plot(freq,lineas.real)
#plt.plot(lambLinea,linea)
#plt.yscale('log',basey=10)


A = trapz(y=linea,x=lambLinea)
#beta = 2*A/lineas.real[0]/np.pi
#beta = A/lineas.real[lambLinea==0]*(3*np.pi+8)/(8*np.pi)
#beta = A/lineas.real[1]*(3*np.pi+8)/(8*np.pi)
#beta = 0.66/(freq[1] * line)
beta = 2*A/linea[lambLinea==0]/np.pi
c = 299792458
vsiniMax = c*beta/1000

#x = np.array([0.0345,0.096,freq[3]+3/4*(freq[4]-freq[3])])
x = np.array([293,664,freq[2]+1.3/4*(freq[3]-freq[2])]) #se hace manualmente.
y = np.array([3.832,7.016,10.174])
yAlta = np.array([4.147,7.303,10.437])
from scipy.stats import linregress
from scipy.optimize import curve_fit

def beta(x,m):
    return m*x


m,b,r,p,sigma = linregress(x,y)
#plt.figure()
#plt.plot(x,m*x+b)
#plt.scatter(x,y)
#print(x/y)
plt.figure()
plt.scatter(x,y)
plt.plot(x, x*m+b)
heh, cvr = curve_fit(beta,x,y)

#plt.plot(x,heh[0]*x)
heh = y/(x*1e4)
vsini1 = 0.660/(line*x[0]*10)*c

vsini2 = heh*c
r

































