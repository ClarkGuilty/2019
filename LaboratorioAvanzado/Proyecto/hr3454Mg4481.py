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

sp1 = np.loadtxt("fhr3454.dat")
#sp2 = np.loadtxt("mhr1544.dat")
sp3 = np.loadtxt("hr3454Reducido.dat")

y1 = sp1[:,1]
y2 = sp1[:,2]
#y3 = sp2[:,1]

lamb = sp1[:,0]
lamb2 = sp3[:,0]
y4 = sp3[:,1]
#plt.plot(lamb,y1*2/(y1[203]+y1[202]))
#plt.plot(lamb,y1)
#plt.plot(lamb2,y4)
conv0 = 2/(y1[202]+y1[203])
#plt.plot(lamb2,y4/conv0)
heLine = 4481
ii = np.where(np.logical_and(lamb2<heLine+6,lamb2>heLine-6))
#plt.plot(lamb2[ii],y4[ii]/conv0)
#plt.title("He 4471 para HR 3453")
#plt.plot(lamb2,y4/conv0)
#plt.plot(lamb,y1)
linea = y4[ii]
#lambLinea = lamb[ii]
plt.plot(lamb2[ii],y4[ii])

from numpy import fft

#deltaSigma = 1.0/(linea.size * (lamb2[1]-lamb2[0]))
#lineas = fft.fft(linea)
#freq = fft.fftfreq(linea.size,deltaSigma)
#iimax = 100
#plt.xlim(0,1130)
#plt.plot(freq*heLine,np.abs(lineas)/np.max(np.abs(lineas)))
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

#print(vel(np.min(lineas.real[iizeros]), heLine), np.argmin(lineas.real[iizeros]))
#print(vel(0.2500000000009095,heLine))

lambLinea = (lamb2[ii] - heLine)/heLine
#linea = linea/linea[lambLinea==0]
linea = np.max(linea) - linea
linea = linea/linea[lambLinea==0]
deltaSigma = 1.0/(linea.size * (lamb[1]-lamb[0]))
lineas = fft.fft(linea)
#freq = fft.fftfreq(linea.size,deltaSigma)
freq = fft.fftfreq(linea.size)
iimax = 100
#plt.xlim(0.008,0.00825)
#plt.xlim(0.00,0.05)
#plt.ylim(-1,1)
#plt.plot(freq,lineas.real/np.max(lineas.real))
#plt.plot(freq,lineas.real)
#plt.plot(lambLinea,linea)
#plt.yscale('log',basey=10)

from scipy.integrate import trapz
A = trapz(y=linea.real,x=lambLinea)
#beta = 2*A/lineas.real[0]/np.pi
#beta = A/lineas.real[lambLinea==0]*(3*np.pi+8)/(8*np.pi)
#beta = A/lineas.real[1]*(3*np.pi+8)/(8*np.pi)
#beta = 0.66/(freq[1] * heLine)
beta = 2*A/linea[lambLinea==0]/np.pi
c = 299792458
vsini = c*beta/1000

#x = np.array([0.0345,0.096,freq[3]+3/4*(freq[4]-freq[3])])
x = np.array([0.0078,0.00810])
y = np.array([3.832,7.016])
from scipy.stats import linregress
from scipy.optimize import curve_fit

def beta(x,m):
    return m*x



#m,b,r,p,sigma = linregress(x,y)
#plt.figure()
#plt.plot(x,m*x+b)
#plt.scatter(x,y)
#print(x/y)

#heh, cvr = curve_fit(beta,x,y)
#plt.plot(x,heh[0]*x)




































