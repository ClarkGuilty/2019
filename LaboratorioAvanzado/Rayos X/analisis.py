#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 18:01:08 2019

@author: Javier Alejandro Acevedo Barroso
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import linregress

#para el LiF
d = 2.014*1e-10 #metros  (110)
hc = 1.986*1e-25
e = 1.602*1e-19

ruidoGrande = np.mean(np.loadtxt("AmbienteFiltroGrande(22022019).txt",delimiter ="	", usecols=1))
ruidoPequeño= 1.8

def l(theta,n=1):
    return 2*d*np.sin(np.pi*theta/180)/n*1e10
#theta = np.arange(3.0,55,0.1)


#parte 4
def average(datos,i):
    return (datos[i-1]+datos[i+1])/2

def lambdaMin(voltaje):
    return hc/(e*voltaje)*1e10
datos = np.loadtxt('parte4(21022019).txt', skiprows = 0, delimiter ="	")
satan = datos[:,11]
satan[30] = average(satan,30)
datos[:,11] = satan

satan = datos[:,11]
satan[35] = average(satan,35)
datos[:,11] = satan

satan = datos[:,9]
satan[158] = average(satan,158)
datos[:,9] = satan

satan = datos[:,10]
satan[174] = average(satan,174)
datos[:,10] = satan



angulos4 = datos[:,0]
nombres = ["13kV","15kV","17kV","19kV","21kV","23kV","25kV","27kV","29kV", "31kV", "33kV", "35kV"]
voltajes = np.array([datos[:,1],datos[:,2],datos[:,3],datos[:,4],datos[:,5],datos[:,6],datos[:,7],datos[:,8],datos[:,9],datos[:,10],datos[:,11],datos[:,12]])
volts = np.linspace(13000,35000,12)
lamdaMin = []
#dictionary = dict(zip(nombres, voltajes))

#for i in range(10):
#    voltajes[-1,i] -= ruidoGrande
#    

#voltajes = voltajes - voltajes[-1]
#voltajes = voltajes


plt.figure(figsize=(8,7))
a=12
f = 24
zeta = 120
lMin = []
print("f es ",f,l(angulos4[f]))
for nombre,voltaje in zip(nombres[:a],voltajes[:a]):
#    voltaje = voltaje/voltaje.sum()
#    voltaje = voltaje/max(voltaje)
    print(nombre,l(angulos4[np.argmax(voltaje[f:zeta])+f]), np.argmax(voltaje[f:zeta])+f)
    plt.plot(l(angulos4[:zeta]),voltaje[:zeta],label=nombre)
#    plt.scatter(lambdaMin(volts), np.linspace(1200,100, 12) ,color = 'black')
    plt.scatter(lambdaMin(volts), np.linspace(100,1700, 12) ,color = 'black')
    lMin.append(l(angulos4[np.argmax(voltaje[f:zeta])+f])/1.5)

lMin = np.array(lMin)
#plt.xlim(3,6)
#plt.xlim(19,23)    
#plt.ylim(0,0.2)
#plt.ylim(-2000,2000) 
plt.ylim(0,2000) 
#plt.ylim(8000,15000)
plt.xlim(.3,.8)
#plt.xlim(.9,1.1)
plt.legend()
plt.xlabel(r"$\lambda$",fontsize=20)
#plt.savefig("Ley de Duane-Hunt.png")





plt.figure(figsize=(8,7))
plt.scatter(volts,lMin)
plt.plot(volts,lambdaMin(volts))


#parte 1


#plt.figure(figsize=(8,7))
#datos1 = np.loadtxt('parte1LiF(21022019).txt', skiprows = 0, delimiter ="	")
#theta1 = datos1[:,0]
#conteos = datos1[:,1]/3
#plt.plot(l(theta1),conteos)
##plt.xlim(0.3,0.8)
##plt.ylim(0,500)
##plt.scatter([lambdaMin(volts[-1])], [20],color='black')
#plt.ylabel('Intensidad (conteos / s)',fontsize=16)
#plt.xlabel(r"Longitud de onda ($\AA$)",fontsize=16)
#plt.title(r"Primer y segundo orden de $K_{\alpha_1}$, $K_{\alpha_2}$ y $K_{\beta}$ en Cu", fontsize = 16)
#plt.savefig("lineasCu.png",dpi=600)



#parte 2
#def intensidadAtenuada( )

    #dAl01 = np.loadtxt('parte2LiFAl(01)(21022019).txt', skiprows = 0, delimiter ="	")
#dAl002 = np.loadtxt('Parte2LiFAl(002)(22022019).txt', skiprows = 0, delimiter ="	")
#dAl004 = np.loadtxt('Parte2LiFAl(004)(22022019).txt', skiprows = 0, delimiter ="	")
#dAl006 = np.loadtxt('parte2LiFAl(006)(21022019).txt', skiprows = 0, delimiter ="	")
#dZn01 = np.loadtxt('parte2LiFZn(01)(21022019).txt', skiprows = 0, delimiter ="	")
#dZn0025 = np.loadtxt('parte2LiFZn(0025)(21022019).txt', skiprows = 0, delimiter ="	")
#dZn0075 = np.loadtxt('parte2LiFZn(0075)(21022019).txt', skiprows = 0, delimiter ="	")
#dNo = np.loadtxt('parte2LiFNoAtenuado(21022019).txt', skiprows = 0, delimiter ="	")
#
#angulos2 = dZn0075[:,0]/2
#angulos2Continuos = np.linspace(min(angulos2),max(angulos2),500)/2
#
#def correccion(N):
#    return N/(1-N*90e-6)
#
#filtros = np.array([dAl002,dAl004,dAl006,dAl01,dZn0025,dZn0075,dZn01,dNo]) - ruidoPequeño 
#filtros = correccion(filtros)
#filtros = filtros/filtros[-1]
#nombres = ['Al 0.002 mm','Al 0.004 mm','Al 0.006 mm','Al 0.01 mm','Zn 0.0025 mm','Zn 0.075 mm','Zn 0.01 mm','Ninguno']




#i=0
#plt.figure(figsize=(8,7))
#for nombre,filtro in zip(nombres[0:],filtros[0:]):
#    plt.plot(l(angulos2),filtro[:,1],label=nombre)
#    print(nombre,filtro.shape, angulos2[np.argmax( filtro[:,1]) ])
#    i+=1
#
#plt.legend(loc=2)
#plt.title("Conteos normalizados contra longitud de onda",fontsize=16)
#plt.savefig('conteosNormalizados.png')
#
#plt.figure(figsize=(8,7))
#for nombre,filtro in zip(nombres[0:],filtros[0:]):
#    plt.plot(angulos2,filtro[:,1]*dNo[:,1],label=nombre)
##    print(nombre,filtro.shape, angulos2[np.argmax( filtro[:,1]) ])
##    i+=1
#
#plt.legend(loc=2)
#plt.title("Conteos contra longitud de onda",fontsize=16)
#plt.savefig('conteos.png')
#
#
#plt.figure(figsize=(7,6))
#for nombre,filtro in zip(nombres[4:],filtros[4:-1]):
#    plt.plot(angulos2,filtro[:,1],label=nombre)
##    print(i,filtro.shape)
##    i+=1
#
#plt.legend()
#plt.savefig('atenuacionZn.png')



#grosorAl = np.array([0.002,0.004,0.006,0.01])
#grosorZn = np.array([0.0025,0.0075,0.01])
#muAl = []
#muZn = []
#
#for i in range(11):
#    intensidades = filtros[:,i][:,1]
##    intensidades = filtros[:,i][:,1]
#    intAl = intensidades[0:4]
#    intZn = intensidades[4:7]
#    intAl[intAl <0] = 0
#    intZn[intZn < 0] = 0    
#    intNo = intensidades[-1]
#    m, b, r, p, sigma = linregress(grosorAl,np.log(intAl))
#    muAl.append(m)
#    m, b, r, p, sigma = linregress(grosorZn,np.log(intZn))
#    muZn.append(m)
    
#plt.figure(figsize=(7,6))
#intensidades = correccion(filtros[:,6][:,1])
#intAl = intensidades[0:4]/intensidades[-1]
#intZn = intensidades[4:7]/intensidades[-1]
#m, b, r, p, sigma = linregress(grosorAl,np.log(intAl))
#plt.plot(grosorAl,m*grosorAl+b, label = 'Ley de atenuación Al, $R^2$ = {:.3}'.format(r*r))
#plt.scatter(grosorAl,np.log(intAl), label = 'Datos Al')
#m, b, r, p, sigma = linregress(grosorZn,np.log(intZn))
#plt.scatter(grosorZn,np.log(intZn), label = 'datos Zn')
#plt.plot(grosorZn,m*grosorZn+b, label = 'Ley de atenuación Zn, $R^2$ = {:.3}'.format(r*r))
#plt.xlabel("Grosor (mm)", fontsize=16)
#plt.ylabel(r'ln $\frac{I}{I_0}$', fontsize=20)
#plt.xlim(0.000,0.011)
#plt.title("Intensidad normalizada vs grosor de la lámina a 6°",fontsize=16)
#plt.legend()
#plt.savefig("intensidad.png")
#
#
#muAl = np.array(muAl)
#muZn = np.array(muZn)
#
#
#
#murhoAl = np.cbrt(muAl/(2.7/1000)*1e1)
##murhoAl = np.cbrt(muAl/26.98)
##murhoAl = muAl
#mAl, bAl, rAl, p, sigmaAl = linregress(murhoAl[1:],l(angulos2[1:])*13)
#
#murhoZn = np.cbrt(muAl/(7.14/1000)*1e1)
##murhoZn = np.cbrt(muZn/(65.38))
##murhoZn = muZn
#a = 1
#mZn, bZn, rZn, p, sigmaZn = linregress(murhoZn[a:],l(angulos2[a:])*30)
#
#plt.figure(figsize=(7,6))
#plt.errorbar(murhoAl[1:],l(angulos2[1:])*13,label='Datos Al',yerr=sigmaAl,fmt='o')
#plt.plot(murhoAl[1:],mAl*murhoAl[1:]+bAl ,label = r"Modelo $R^2$ = {:.3}".format(rAl*rAl))
#plt.errorbar(murhoZn[a:],l(angulos2[a:])*30,label='Datos Zn',yerr=sigmaZn,fmt='o')
#plt.plot(murhoZn[a:],mZn*murhoZn[a:]+bZn ,label = r"Modelo $R^2$ = {:.3}".format(rZn*rZn))
#plt.ylabel(r'$\lambda Z$ ($\AA$)', fontsize=16)
#plt.xlabel(r'($\frac{\mu}{\rho}$)$^{1/3}$ $(\frac{mm^2}{g}$)$^{1/3}$', fontsize=24)
#plt.title("Longitud de onda vs raíz cúbica de la atenuación por densidad",fontsize=14)
#plt.legend()
#plt.savefig("LvsMu.png")
#
#
#







































