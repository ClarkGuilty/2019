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


#f = fits.open("_hr1544_20190227_035_full.fit")
#specdata = f[0]
#
#
#from specutils import Spectrum1D
#lamb = 10**specdata['loglam'] * u.AA # doctest: +REMOTE_DATA
#flux = specdata['flux'] * 10**-17 * u.Unit('erg cm-2 s-1 AA-1') # doctest: +REMOTE_DATA
#spec = Spectrum1D(spectral_axis=lamb, flux=flux) # doctest: +REMOTE_DATA

sp1 = np.loadtxt("fhr1544.dat")
sp2 = np.loadtxt("mhr1544.dat")
sp3 = np.loadtxt("sirioReducido.dat")

y1 = sp1[:,1]
y2 = sp1[:,2]
y3 = sp2[:,1]

lamb = sp1[:,0]
lamb2 = sp3[:,0]
y4 = sp3[:,1]


#plt.plot(lamb,y1*2/(y1[203]+y1[202]))
#plt.plot(lamb,y1)
#plt.plot(lamb2,y4)
conv0 = 2/(y1[202]+y1[203])
line = 4861
ii = np.where(np.logical_and(lamb2<line+7,lamb2>line-7))

#plt.plot(lamb2,y4/conv0*1e-7)
plt.title("Mg 4481 en HR1544 transformada" ,fontsize=20)
#plt.ylabel("Flujo [J cm$^{-2}$ $\AA^{-1}$ s$^{-1} \ 10^{16}]$", fontsize=18)
#plt.xlabel(r'Longitud de onda [$\AA$]', fontsize=18)
linea = y4[ii]/conv0
lambLinea = lamb2[ii]
lambLinea = (lamb2[ii] - line)/line
#plt.plot(lamb2[ii],y4[ii]/conv0)
#plt.xlim(min(lamb2),6700)
linea = np.max(linea) - linea
linea = linea/np.max(linea)
plt.plot(lambLinea,linea)
plt.ylabel("Intensidad relativa [$F/F_{\lambda = 4481}$]", fontsize=18)
plt.xlabel(r'($\lambda - \lambda_m)/\lambda_m$', fontsize=18)
#plt.xlabel(r'$\frac{\lambda - \lambda_m}{\lambda_m}$', fontsize=18)
plt.savefig("HR1544Mg4481B.png", dpi = 1000, bbox_inches='tight')
#plt.plot(lamb,y3)

from numpy import fft
from scipy.integrate import trapz
#deltaSigma = 1.0/(linea.size * (lamb2[1]-lamb2[0]))
#lineas = fft.fft(linea)
#freq = fft.fftfreq(linea.size,deltaSigma)
#iimax = 100
#plt.xlim(0,1130)
#plt.plot(freq*line,np.abs(lineas)/np.max(np.abs(lineas)))
#plt.yscale('log',basey=10)
#plt.xscale('log',basex=10)
#iizeros = np.where(np.logical_and(freq > 0,True))
#print(iizeros)
#plt.plot(freq[iizeros],lineas.real[iizeros])
#plt.xlim
    

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def vel(sigma1, longOnda):
    return 0.66 * 299792458/(longOnda)/sigma1

#print(vel(np.min(lineas.real[iizeros]), line), np.argmin(lineas.real[iizeros]))
#print(vel(0.2500000000009095,line))

lambLinea = (lamb2[ii] - line)/line
#linea = linea/linea[lambLinea==0]
#linea = np.max(linea) - linea
linea = linea/np.max(linea)
deltaSigma = 1.0/(linea.size * (lamb2[1]-lamb2[0]))
lineas = fft.fft(linea)
#freq = fft.fftfreq(linea.size,deltaSigma)
freq = fft.fftfreq(linea.size, d = lambLinea[1]-lambLinea[0])
iimax = 100
#plt.xlim(0.008,0.00825)
plt.figure()
#plt.xlim(0.005,0.01)
#plt.xlim(-0,0.02)
#plt.ylim(-0.05,0.1)
plt.xlabel("Frecuencias [hz]", fontsize=18)
plt.ylabel("FFT normalizada", fontsize=18)
plt.title("FFT de H 4861 en HR1544", fontsize=18)
plt.plot(freq,lineas.real/np.max(lineas.real))
plt.scatter(freq,lineas.real/np.max(lineas.real), color = 'orange')
plt.xlim(0,1000)
plt.savefig("HR1544Mg4481.png", dpi = 1000, transparent=True,bbox_inches='tight')
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
x = np.array([0.0030,freq[2],freq[2]+1.3/4*(freq[3]-freq[2])])
y = np.array([3.832,7.016,10.174])
yAlta = np.array([4.147,7.303,10.437,13.569])
from scipy.stats import linregress
from scipy.optimize import curve_fit

def beta(x,m):
    return m*x

print(vel(x[0],line))

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

vsini1 = 0.660/(line*x[0])


































