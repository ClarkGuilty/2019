#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:40:41 2019

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import linregress, mode
from pymc3 import *



muN = 5.050783699*10**(-27) # J/ T
#hbar = 1.054571800*10**(-34) # J s
hbar = 6.626070*10**(-34) # J s

def milis(x, pos):
    'The two args are the value and tick position'
    return '%1.1f' % (x * 1e3)


def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1f' % (x * 1e-6)

formatterY = FuncFormatter(millions)
formatterX = FuncFormatter(milis)

Graficar = False
fig,ax = plt.subplots(figsize=(8,6))
ax.xaxis.set_major_formatter(formatterX)
ax.yaxis.set_major_formatter(formatterY)
plt.title("Frecuencia vs campo para diferentes muestras",fontsize=15)


def fitGraficar(archivo,titulo, material):

    datos = np.loadtxt(archivo)
    y = datos[:,0]*1e6
    ey = datos[:,2]*1e6
    x = datos[:,1]*1e-2
    ex = datos[:,3]*1e-2
    plt.errorbar(x,y,xerr=ex,yerr=ey,fmt='o', label = 'Datos '+material)
    m, b, r, p, std = linregress(x,y)
    xtest = np.linspace(0,np.max(x))
#    xtest = np.linspace(np.min(x),np.max(x))
    #plt.plot(xtest, m*xtest+b, label = r'Regresion $R^2$ = {:.3}'.format(r**2))
    plt.plot(xtest, m*xtest+b, label = r'Regresion bayesiana '+material)
    plt.ylabel('Frecuencia (MHz)', fontsize = 14)
    plt.xlabel(r'$B_0$ (mT)', fontsize = 14)
    plt.legend()
#    plt.title(titulo, fontsize = 16)
    m, b, r, p, std = linregress(x, y) #Regresion de w = gamma B0
    print(m*1e-6,std*1e-6)
#    return np.array([gA(m), gA(std*3)])
    return [m*1e-6,3*std*1e-6]

def gA(gammaN):
    return gammaN*hbar/muN

def lat(g,letra):
#    print(g)
    print("g_{"+letra+"}} & ${0:.4} \\pm {1:.3}$ \\\\".format(g[0],g[1]))

def perc(calculado,real):
    return np.abs(real-calculado)/real*100
#    
gH = fitGraficar('parte2Glicerina(07022018).dat', "Campo contra frecuencia - Glicerina (H)", "Glicerina (H)")
gF1 = fitGraficar('parte2CremaDental(08022018).dat', "Campo contra frecuencia - Crema Dental (F)","Crema Dental (F)")
gF2 = fitGraficar('parte2Teflon(08022018).dat', "Campo contra frecuencia - Teflón (F)","Teflón (F)")
#plt.plot(xtest*1e-2, 1e6*xtest*trace['x'].mean()+1e6*trace['Intercept'].mean(), label='mejor ajuste bayesiano', lw=3., c='y')

plt.savefig("Regresiones.png",dpi = 600)
#print(gH,gF1,gF2)
lat(gH,'H')
lat(gF1,'F')
lat(gF2,'F')
print(perc(17.55,5.588))
print(perc(9.725,5.2567))
print(perc(9.939,5.2567))
#
#    
##Análisis bayesiano de los datos.
#
#data = np.loadtxt('parte2Glicerina(07022018).dat')
data = np.loadtxt('parte2CremaDental(08022018).dat')
#data = np.loadtxt('parte2Teflon(08022018).dat')
x = data[:,1]
y = data[:,0]
sigma = data[:,3]*4
xtest = np.linspace(np.min(x),np.max(x))

def inv(eme):
    return eme**-1

#with Model() as model: # model specifications in PyMC3 are wrapped in a with-statement
#    # Define priors
#    intercept = Normal('Intercept', 0, sd=20)
#    x_coeff = Normal('x', 0, sd=20)
#
#    # Define likelihood
#    likelihood = Normal('y', mu=intercept + x_coeff * x,
#                        sd=sigma, observed=y)
#
#    # Inference!
#    trace = sample(5000, njobs=8) # draw 3000 posterior samples using NUTS sampling



#plt.figure(figsize=(7, 7))
#traceplot(trace[100:])
#plt.tight_layout();
#
#
#plt.figure(figsize=(7, 7))
#plt.plot(x, y, 'x', label='data')
#plt.plot(xtest, xtest*trace['x'].mean()+trace['Intercept'].mean(), label='mejor ajuste bayesiano', lw=3., c='y')
#plt.plot(xtest, xtest*np.median(trace[1000:].x)+np.median(trace[1000:].Intercept), label='regresiones alternativas', lw=1.5, c='b')
#
#
#plt.title('Posterior predictive regression lines')
#plt.legend(loc=0)
#plt.ylabel('Frecuencia (MHz)', fontsize = 12)
#plt.xlabel(r'$B_0$ (mT)', fontsize = 12)
#
#
#
#print("El g_F del Teflón es {:.2f} \pm {:.2f}".format(gA(trace['x'].mean()**-1)*1e8, gA(trace['Intercept'].mean()*2)*1e8))


















