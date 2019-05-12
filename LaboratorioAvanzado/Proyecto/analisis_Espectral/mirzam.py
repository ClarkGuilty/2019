#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:10:21 2019

@author: Javier Alejandro Acevedo Barroso
"""


import numpy as np
import matplotlib.pyplot as plt

espectre = np.loadtxt("MirzamProm.dat")
star = 'Mirzam'

lamb = espectre[:,0]
y4 = espectre[:,1]

dpi = 300
#plt.plot(lamb,y4)
#conv0 = 2/(y1[202]+y1[203])
element = 'H'
line = 4341.2 #
#line = 4861 #
#line = 6564 #
lineS = str(line)
ii = np.where(np.logical_and(lamb<line+5,lamb>line-5))
iii = np.where(np.logical_and(lamb<line+15,lamb>line-15))

plt.plot(lamb[iii],y4[iii])
plt.figure()
#plt.plot(lamb,y4/conv0*1e-7)
plt.title(element+" "+lineS+" en "+star+" transformada" ,fontsize=20)
#plt.ylabel("Flujo [J cm$^{-2}$ $\AA^{-1}$ s$^{-1} \ 10^{16}]$", fontsize=18)
#plt.xlabel(r'Longitud de onda [$\AA$]', fontsize=18)
linea = y4[ii]
lambLinea = lamb[ii]
lambLinea = (lamb[ii] - line)/line
linea = np.max(linea) - linea
linea = linea/np.max(linea)
plt.plot(lambLinea,linea)
plt.ylabel("Intensidad relativa [$F/F_{\lambda = "+lineS+"}$]", fontsize=18)
plt.xlabel(r'($\lambda - \lambda_m)/\lambda_m$', fontsize=18)
#plt.xlabel(r'$\frac{\lambda - \lambda_m}{\lambda_m}$', fontsize=18)
#plt.savefig("HR1544Mg4481B.png", dpi = dpi, bbox_inches='tight')


from numpy import fft
from scipy.fftpack import dct
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from scipy.optimize import ridder

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def recta(j):
    yy2 = lineas[j+1]
    xx2 = freq[j+1]
    yy1 = lineas[j]
    xx1 = freq[j]
    mm = (yy2-yy1)/(xx2-xx1)
    bb = yy2-mm*xx2
    return  -bb/mm

linea = linea/np.max(linea)
deltaSigma = 1.0/(linea.size * (lamb[1]-lamb[0]))
lineas = dct(np.abs(linea))
#freq = fft.fftfreq(linea.size,deltaSigma)
freq1 = fft.fftfreq(linea.size, d = lambLinea[1]-lambLinea[0]) #El otro paper sí utiliza el espaciado.
freq = fft.fftfreq(linea.size)*2*np.pi #Interesantemente, en el paper original no tienen encuenta el espaciado. u = 2*pi*k
freqTest = np.linspace(np.min(freq),np.max(freq),1000)
iimax = 100
funcionTestFourier = interp1d(freq,lineas,kind='quadratic')
#plt.xlim(0,1000)
plt.figure()
plt.xlabel("Frecuencias [hz]", fontsize=18)
plt.ylabel("FFT normalizada", fontsize=18)
plt.title("FFT de H 4861 en "+star, fontsize=18)
#plt.xlim(3500,4500)
#plt.ylim(-0.025,0.0251)
plt.xlim(0,0.43)

plt.plot(freq,np.zeros_like(freq))
plt.plot(freq,lineas.real/np.max(lineas.real),color = 'r')
plt.scatter(freq,lineas.real/np.max(lineas.real), color = 'orange')
plt.plot(freqTest,funcionTestFourier(freqTest)/np.max(lineas),color = 'blue')

A = trapz(y=linea,x=lambLinea)
beta = 2*A/linea[lambLinea==0]/np.pi
c = 299792458
vsiniMax = c*beta*1e-3

#x = np.array([0.0345,0.096,freq[3]+3/4*(freq[4]-freq[3])])
#x = np.array([recta(2),recta(4),recta(6)]) #se hace manualmente.
x = np.array([ridder(funcionTestFourier,freq[0],freq[2]),ridder(funcionTestFourier,freq[2],freq[3]),ridder(funcionTestFourier,freq[4],freq[10])]) #se hace manualmente.
funcionTestFourier = interp1d(freq,lineas)

yAlta = np.array([3.832,7.016,10.174])
y = np.array([4.147,7.303,10.437])
from scipy.stats import linregress
from scipy.optimize import curve_fit



m,b,r,p,sigma = linregress(x,y)

plt.figure()
plt.scatter(x,y)
plt.plot(x, x*m+b)

def ff(x,mmm):    return x*mmm
heh, cvr = curve_fit(ff,x,y)
#vsini2 = heh.mean()*c*2.0571e-5/1000
vsini2 = heh.mean()*c*1e-6/1000 , np.sqrt(cvr.mean())*c*1e-6/1000
#plt.plot(x,heh[0]*x)
heh = y/(x)
funcionRealFourier = interp1d(freq1,lineas,kind='quadratic')
vsini3 = np.array([heh.mean()-heh.std(),heh.mean()+heh.std()])*c*1e-6/1000
vsini4 = heh.mean()*1e-9*c, heh.std()*c*1e-9
vsini1 = 0.660/(line*ridder(funcionTestFourier,freq[0],freq[2])*10)*c*1e-4 #parece ser que el factor es 1e-3


def PlanB(w):
    return np.trapz(y=lineas*np.cos(w*lambLinea),x=lambLinea)

plt.figure()
PlanB(freq)
planB = np.zeros_like(freq1)
for i in range(len(freq1)):
    planB[i] = PlanB(freq1[i])

plt.plot(freq,planB)























