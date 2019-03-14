#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:39:43 2019

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction


def gF(S,L,J,I,F):
#    print("gj = ", gJ(J,L,S))
    return gJ(J,L,S)*(F*(F+1)+J*(J+1)-I*(I+1))/(2*F*(F+1))

def gJ(J,L,S):
    return 1 + (J*(J+1) + S*(S+1) - L*(L+1))/(2*J*(J+1))

def p(nS):
    return (nS-1)/2

#Estados posibles

#Retorna los J_Total para cada par (J1,J2).
def estados(J1, J2):
    rta1 = []
    rta2 = []
    actual = J1+J2
    while(actual >= np.abs(J2-J1) ):
        rta1.append(Fraction(actual))
        rta2.append(actual)
        actual -=1
#    return rta1
    return np.array(rta2)


A = {}
A["3/2"] = 3035.73 #Mhz * h
A["5/2"] = 6834.68
muI = {}
muI["5/2"] = 1.35303
muI["3/2"] = 2.75124

nombres = {}
nombres["3/2"]="87 Rb"
nombres["5/2"]="85 Rb"

pte = 1836.15267389
muB = 9.274009994*1e-24
B = 4.58e-4
h = 6.62607015*1e-34

def Energy(S,L,J,I,F,M):
    a = A[str(Fraction(I).limit_denominator())]*h*1e6
    gj = gJ(J,L,S)
#    gf = gF(S,L,J,I,F)
    gi = -muI[str(Fraction(I).limit_denominator())]/pte/I
#    x = (gj*muB-gi*muB/pte)*B/a
    x = (gj-gi)*muB*B/a
#    print("x es ",x, gi,gj,J)
#    return (-a*M/(gj/gi-1.0)*x + J*a*np.sqrt(1 + 4*M/(2*I+1)*x + x*x)-a/(2*(2*I+1)))/h*1e-6 #Versión guia
#    return (-M*B*(muB/pte)*gi + J*a*np.sqrt(1 + 2*M/(I+0.5)*x + x*x)-a/(2*(2*I+1)))/h*1e-6
    y = M*B*muB*gi-a/(2*(2*I+1))
    if(M == -I-1/2):
        return (y+J*a*(1-x))*1e6/h
    return (M*B*muB*gi + J*a*np.sqrt(1 + 2*M/(I+0.5)*x + x*x)-a/(2*(2*I+1)))*1e-6/h
    

#Retorna la frecuencia de transición por zeeman débil en Mhz
def Edebil(S,L,J,I,F,M):
    return gF(S,L,J,I,F)*muB*M*B/h*1e-6

J = 1/2
S = 1/2
print("Primer punto")
print("S","L","J","I","F", "g_F","g_J")
for L in [0,1]:
    print("átomo {:d}".format(L+1))
    for I in [3/2,5/2]:
        for F in estados(J,I):
            print(S,L,J,I,F, Fraction(gF(S,L,J,I,F)).limit_denominator(),Fraction(gJ(S,L,J)).limit_denominator())            



print("Segundo punto")
F = 3
print("S","L","J","I","F", "M", "f[Mhz]")
for L in [0,1]:
    print("átomo {:d}".format(L+1))
    for I in [3/2,5/2]:
        for M in np.arange(0,F+1):
#            print(S,L,J,I,F,M,"E {:4.3f} Mhz".format(Energy(S,L,J,I,F,M)))
            print(S,L,J,I,F,M,"{:4.3f}".format(Edebil(S,L,J,I,F,M)))
 
           

#punto 3 
def f(S,L,J,I,F,M,x):
    a = A[str(Fraction(I).limit_denominator())]*h*1e6
    gj = gJ(J,L,S)
#    gf = gF(S,L,J,I,F)
    gi = -muI[str(Fraction(I).limit_denominator())]/pte/I
#    x = (gj*muB-gi*muB/pte)*B/a
#    print(J*a*np.sqrt(1 + 4*M/(2*I+1)*x + x*x))
#    return (-a*M/(gj/gi-1.0)*x + J*a*np.sqrt(1 + 4*M/(2*I+1)*x + x*x)-a/(2*(2*I+1)))/h
    if(M == -I-1/2):
        return (muB*gi*M*B+ J*a*(1-x)-a/(2*(2*I+1)))/h
    return (muB*gi*M*B+ J*a*np.sqrt(1 + 4*M/(2*I+1)*x + x*x)-a/(2*(2*I+1)))/h

            
x = np.linspace(0,3,100)
for I in [3/2,5/2]:
#    print("Isotopo ",nombres[str(Fraction(I).limit_denominator())] )
    plt.figure()
    plt.title("Frecuencia de la transición {:s}".format(nombres[str(Fraction(I).limit_denominator())]),fontsize=16)
    plt.xlabel("Intensidad de campo x",fontsize=16)
    plt.ylabel("Frecuencia del foton [Hz]",fontsize=16)
    for J in [-1/2,1/2]:
        for F in [J+I,J-I]: 
            for M in np.arange(-F,F+1):
                plt.plot(x,f(S,L,J,I,F,M,x), label ="F={:d},M={:d}".format(int(F),int(M)    ))
                
                
    plt.legend()
    plt.savefig('preinforme '+nombres[str(Fraction(I))]+'.png',dpi=600)


I = 3/2
F = 2
M = -2

#plt.plot(x,f(S,L,J,I,F,M,x), label ="{:.1f} {:.1f} {:.1f} {:.1f}".format(J,I,F,M    ))











