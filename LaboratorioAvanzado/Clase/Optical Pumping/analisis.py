#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 18:21:01 2019

@author: Javier Alejandro Acevedo Barroso
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


datos = np.loadtxt('datosZeemanDebil(07032019).txt', skiprows=1)
freq = datos[:,0]
realFreq = np.array([100.5,109.8,118.5,127.0,137,147.8,157,167.8,175,190])
freq=realFreq
p0 = datos[:,1]/1000
p1 = datos[:,2]/1000 -p0
p2 = datos[:,3]/1000 -p0

m1, b1, r1, pp1, sigma1 = linregress(p1,freq)
m2, b2, r2, pp2, sigma2s = linregress(p2[:-1],freq[:-1])

nombres = ["Rb 87", "Rb 85"]
bes = [b1,b2]
eme = [m1,m2]
gF = [0.5,1./3]

h = 6.62607e-34
muB = 5.05078324e-27

def B(I,isotopo = 0):
    return h*(eme[isotopo]*I + bes[isotopo])*1000/(gF[isotopo] * muB)*1e3

Iprueba = np.linspace(0,1,100)
Iprueba2 = np.linspace(np.min(p1),1,100)

plt.figure(figsize=(7,6))

plt.scatter(p2[:-1],freq[:-1], label = "datos Rb 85")
plt.scatter(p1,freq, label = "datos Rb 87")
plt.plot(Iprueba,m2*Iprueba+b2, label = r"Ajuste Rb 85 $R^2$ = {:.3}".format(r2*r2))
plt.plot(Iprueba,m1*Iprueba+b1, label = r"Ajuste Rb 87 $R^2$ = {:.3}".format(r1*r1))
plt.xlim(0,1)
plt.ylim(0,200)
plt.xlabel("Corriente (mA)", fontsize=16)
plt.ylabel("Frecuencia (kHz)", fontsize=16)
plt.title("Frecuencia de la transición vs corriente del campo",fontsize=16)
plt.legend()
plt.savefig("FvsI.png",dpi=600)


plt.figure(figsize=(7,6))
plt.plot(Iprueba2,B(Iprueba2,isotopo=1), label = "Campo a partir de {}".format(nombres[1]), color = 'blue')
#plt.plot(Iprueba*1e-3,B(Iprueba,isotopo=0), label = "Campo a partir de {}".format(nombres[0]), color = 'orange')
plt.scatter(Iprueba2,B(Iprueba2,isotopo=0),marker = '+', label = "Campo a partir de {}".format(nombres[0]), color = 'orange')

plt.title(r"Calibración de $B$ de barrido para ambos isótopos", fontsize=16)
plt.ylabel("Campo magnético de barrido (mT)",fontsize=16)
plt.xlabel("Corriente (A)",fontsize=16)
plt.legend()
plt.savefig("BvsI.png",dpi=600)




m1, b1, r1, pp1, sigma1 = linregress(Iprueba2[0:-1:5],B(Iprueba2[0:-1:5],isotopo=0))
m2, b2, r2, pp2, sigma2 = linregress(Iprueba2[0:-1:5],B(Iprueba2[0:-1:5],isotopo=1))
#print("B = {:.3f} I + ({:.3f}) para {}".format(m1,b1,nombres[0]))
#print("B = {:.3f} I + ({:.3f}) para {}".format(m2,b2,nombres[1]))

print("B = {:.3f} I + ({:.3f}) para {}".format(m1,b1,nombres[0]))
print("B = {:.3f} I + ({:.3f}) para {}".format(m2,b2,nombres[1]))
print("B = {:.3f} I + ({:.3f})".format(0.5*(m1+m2),0.5*(b1+b2)))










