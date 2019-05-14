#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:41:47 2019

@author: Javier Alejandro Acevedo Barroso
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

si = np.loadtxt("sirioDespues.dat")
lambA = si[:,0]
si = si[:,1]
no = np.loadtxt("sirioAntes.dat")
lambB = no[:,0]
no = no[:,1]

plt.figure(figsize=(12,6))
iiA = np.logical_and(lambA < 6700,lambA > 4300)
iiB = np.logical_and(lambB < 6700,lambB > 4300)
plt.plot(lambA[iiA],si[iiA]/np.max(si[iiA]), label = "Con corrección de flujo",linewidth=0.7)
plt.plot(lambB[iiB],no[iiB]/np.max(no[iiB]), label = "Sin corrección de flujo", linewidth = 0.7)


#plt.ylim(0,0.01)
plt.xlim(min(lambA), 6700)
plt.title("Calibración de flujo de Sirio", fontsize=28)
plt.xlabel(r'Longitud de onda [$\AA$]', fontsize=22)
plt.ylabel("Intensidad relativa", fontsize=22)
plt.legend(prop={'size': 20})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("calibracionSirio.png", dpi = 600, transparent=True,bbox_inches='tight')
