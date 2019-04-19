#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:41:47 2019

@author: Javier Alejandro Acevedo Barroso
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

si = np.loadtxt("1544SI.dat")
lamb = si[:,0]
si = si[:,1]
no = np.loadtxt("1544NO.dat")[:,1]

Asi = trapz(si,lamb)
Ano = trapz(no,lamb)

ii = np.logical_and(lamb < 6700,lamb > 4300)
#plt.plot(lamb,no/Ano)
plt.plot(lamb[ii],no[ii]/np.max(no[ii]), label = "Sin corrección de flujo")
#plt.plot(lamb,si/Asi*8)
plt.plot(lamb[ii],si[ii]*5/np.max(no[ii]), label = "Con corrección de flujo")

#plt.ylim(0,0.01)
plt.xlim(min(lamb), 6700)
plt.title("Calibración de flujo de HR1544", fontsize=18)
plt.xlabel(r'Longitud de onda [$\AA$]', fontsize=18)
plt.ylabel("Intensidad relativa", fontsize=18)
plt.legend()
plt.savefig("hr1544Flujo.png", dpi = 1000, transparent=True,bbox_inches='tight')