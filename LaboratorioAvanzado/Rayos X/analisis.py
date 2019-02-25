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


#para el LiF
d = 2.014*1e-10 #metros  (110)

#theta = np.arange(3.0,55,0.1)


#parte 4
#datos = np.loadtxt('parte4(21022019).txt', skiprows = 0, delimiter ="	")
#satan = datos[:,11]
#satan[30] = (satan[29]+satan[31])/2
#
#angulos4 = datos[:,0]
#nombres = ["13kV","15kV","17kV","19kV","21kV","23kV","25kV","27kV","29kV", "31kV", "33kV", "35kV"]
#voltajes = [datos[:,1],datos[:,2],datos[:,3],datos[:,4],datos[:,5],datos[:,6],datos[:,7],datos[:,8],datos[:,9],datos[:,10],datos[:,11],datos[:,12]]
#dictionary = dict(zip(nombres, voltajes))
#
#
#
#plt.figure(figsize=(10,10))
#for nombre,voltaje in zip(nombres,voltajes):
##    voltaje = voltaje/voltaje.sum()
#    plt.plot(angulos4,voltaje,label=nombre)
#
#plt.xlim(3,6)
##plt.xlim(19,23)    
##plt.ylim(0,0.02)
#plt.ylim(0,2000)
#plt.legend()
#plt.savefig("Ley de Duane-Hunt.png")

#parte 1


#plt.figure(figsize=(10,8))
#datos1 = np.loadtxt('parte1LiF(21022019).txt', skiprows = 4, delimiter ="	")
#theta1 = datos1[:,0]
#conteos = datos1[:,1]
#plt.plot(theta1,conteos)
#
#plt.savefig("lineasCu.png")



#parte 2
#def intensidadAtenuada( )

dAl01 = np.loadtxt('parte2LiFAl(01)(21022019).txt', skiprows = 0, delimiter ="	")
dAl002 = np.loadtxt('Parte2LiFAl(002)(22022019).txt', skiprows = 0, delimiter ="	")
dAl004 = np.loadtxt('Parte2LiFAl(004)(22022019).txt', skiprows = 0, delimiter ="	")
dAl006 = np.loadtxt('parte2LiFAl(006)(21022019).txt', skiprows = 0, delimiter ="	")
dZn01 = np.loadtxt('parte2LiFZn(01)(21022019).txt', skiprows = 0, delimiter ="	")
dZn0025 = np.loadtxt('parte2LiFZn(0025)(21022019).txt', skiprows = 0, delimiter ="	")
dZn0075 = np.loadtxt('parte2LiFZn(0075)(21022019).txt', skiprows = 0, delimiter ="	")
dNo = np.loadtxt('parte2LiFNoAtenuado(21022019).txt', skiprows = 0, delimiter ="	")

angulos2 = dZn0075[:,0]/2
angulos2Continuos = np.linspace(min(angulos2),max(angulos2),500)/2

filtros = np.array([dAl002,dAl004,dAl006,dAl01,dZn0025,dZn0075,dZn01,dNo])
nombres = ['Al 0.002 mm','Al 0.004 mm','Al 0.006 mm','Al 0.01 mm','Zn 0.0025 mm','Zn 0.075 mm','Zn 0.01 mm','Ninguno']

#i=0
#plt.figure(figsize=(10,8))
#for nombre,filtro in zip(nombres[0:],filtros[0:]):
#    plt.plot(angulos2,filtro[:,1],label=nombre)
#    print(nombre,filtro.shape, angulos2[np.argmax( filtro[:,1]) ])
#    i+=1
#
##plt.plot(angulos2,dNo[:,1],label=nombres[-1])
#plt.legend()
#plt.savefig('atenuacionAl.png')


#plt.figure(figsize=(10,10))
#for nombre,filtro in zip(nombres[3:],filtros[3:]):
#    plt.plot(angulos2,filtro[:,1],label=nombre)
#    print(i,filtro.shape)
#    i+=1

#plt.legend()
#plt.savefig('atenuacionZn.png')

intensidades = filtros[:,5][:,1] #(se toma la intensidad en el angulo de 11°)
intAl = intensidades[0:4]
intZn = intensidades[4:7]
intNo = intensidades[-1]
grosorAl = np.array([0.002,0.004,0.006,0.01])
grosorZn = np.array([0.0025,0.0075,0.01])

plt.plot(grosorAl,intAl)
plt.plot(grosorZn,intZn)
























