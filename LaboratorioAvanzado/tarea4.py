#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 20:27:21 2019

@author: Javier Alejandro Acevedo Barroso
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import linregress
from scipy.special import factorial
from numpy import random

def stdf(arr):
      return np.sqrt(np.sum((arr-np.mean(arr))**2)/(arr.size-1))

def gauss(x,mean,std):
      return np.sqrt(1./(2*std*std*np.pi))*np.exp(-(x-mean)**2/(2*std**2))

def err(xi,xf,mean,std):
      return quad(gauss,xi,xf,args=(mean,std))[0]

def poisson(N,meanN):
      return np.exp(-meanN)*meanN**N/factorial(N)


#def z1(a):
#      return 2*a
#
#def z2(a):
#      return a/2
#
#def z3(a):
#      return (a-1)/(a+1)
#
#def z4(a):
#      return a*a/(A-2)
#
#def z5(a):
#      return np.math.asin(1/a)
#
#def z6(a):
#      return np.sqrt(a)
#
#def z7(a):
#      return np.log(1/np.sqrt(a))
#
#def z8(a):
#      return np.exp(a**2)
#
#def z9(a):
#      return a + np.sqrt(1/a)
#
#def z10(a):
#      return 10**a
#
#print('punto 1:')
#
#A = 9.274
#da = 0.005
#i = 0
#for f in [z1,z2,z3,z4,z5,z6,z7,z8,z9,z10]:
#      print("Para Z_{:d} se tuvo {:.3f} \pm {:.3f}".format(i,f(A), f(A+da)-f(A)))
#      i+=1
      
      
#print('punto 2:')
#A = 12.3
#dA = 0.4
#B = 5.6
#dB = 0.8
#C =89.0
#dC = 0.2
#
#def mZ1(a,b):
#      return a+b
#
#def mZ2(a,b):
#      return a-b
#
#def mZ3(a,b):
#      return (a-b)/(a+b)
#
#def mZ4(a,b,c):
#      return a*b/c
#
#def mZ5(a,b):
#      return np.math.asin(b/a)
#
#def mZ6(a,b,c):
#      return a*b*b*c*c*c
#
#def mZ7(a,b,c):
#      return np.log(a*b*c)
#
#def mZ8(a,b,c):
#      return a*b*c
#
#def mZ9(a,b,c):
#      return a+np.math.tan(b/c)
#
#def mZ10(a,b,c):
#      return a*b*c
#
def errorAB(f,a,da,b,db):
      total = ((f(a,b) - f(a+da,b))*da)**2 + ((f(a,b) - f(a,b+db))*db)**2
      return np.sqrt(total)
#      
#def errorABC(f,a,da,b,db,c,dc):
#      total = ((f(a,b,c) - f(a+da,b,c))*da)**2 + ((f(a,b,c) - f(a,b+db,c))*db)**2 + ((f(a,b,c) - f(a,b,c+dc))*dc)**2
#      return np.sqrt(total)
#      
#
#for i, f in zip([1,2,3,5], [mZ1,mZ2,mZ3,mZ5]):
#      print(i,f(A,B),errorAB(f,A,dA,B,dB))
#      
#for i, f in zip([4,6,7,9], [mZ4,mZ6,mZ7,mZ9]):
#      print(i,f(A,B,C),errorABC(f,A,dA,B,dB,C,dC))

#print('punto 5:')
#ti = 25.0 * np.pi / 180
#dti = 0.1 * np.pi / 180
#n = 1.54
#dn = 0.01
#
#def snell(n,thetai):
#      return np.math.asin(np.sin(thetai) / n)
#
#print(snell(n,ti)*180/np.pi, errorAB(snell,n,dn,ti,dti)*180/np.pi)


#print('punto 10:')
#C = np.array([3.03,2.99,2.99,3.00,3.05,2.97])
#dC = np.array([0.04,0.03, 0.02,0.05,0.04, 0.02])
#
#xCE = np.sum(C/dC/dC)/np.sum(1/dC**2)
#aCE = np.sqrt(1 / np.sum(1/dC**2))
#print("caso 0 {:.2f} \pm {:.2f}".format(xCE,aCE))
#
#C = np.array([3.03,2.99,2.99,3.00,3.05,2.97,3.00])
#dC = np.array([0.04,0.03, 0.02,0.05,0.04, 0.02,0.3])
#
##print(np.mean(C), np.mean(dC))
#
#xCE = np.sum(C/dC/dC)/np.sum(1/dC**2)
#aCE = np.sqrt(1 / np.sum(1/dC**2))
#print("caso 1 {:.2f} \pm {:.2f}".format(xCE,aCE))
#
#C = np.array([3.03,2.99,2.99,3.00,3.05,2.97,3.00,4.01])
#dC = np.array([0.04,0.03, 0.02,0.05,0.04, 0.02,0.3,0.01])
#
##print(np.mean(C), np.mean(dC))
#
#xCE = np.sum(C/dC/dC)/np.sum(1/dC**2)
#aCE = np.sqrt(1 / np.sum(1/dC**2))
#print("caso 2 {:.2f} \pm {:.2f}".format(xCE,aCE))


print('tarea 5, punto 3:')
I = np.arange(10,100,10)
V = np.array([0.98,1.98,2.98,3.97,4.95,5.95,6.93,7.93,8.91])

def deltaReg(x,y):
      return x.size*np.sum(x*x)-np.sum(x)**2

def cReg(x,y):
      return (np.sum(x*x)*np.sum(y)-np.sum(x)*np.sum(x*y) )/deltaReg(x,y)

def mReg(x,y):
      return (x.size* np.sum(x*y) - np.sum(x)*np.sum(y))/deltaReg(x,y)

def errCU(x,y):
      c = cReg(x,y)
      m = mReg(x,y)
      return np.sqrt( np.sum( ( y-m*x-c )**2 )/(x.size-2))

def errC(x,y):
      return errCU(x,y)*np.sqrt(np.sum(x*x)/deltaReg(x,y))

def errM(x,y):
      return errCU(x,y)*np.sqrt(x.size/deltaReg(x,y))


dV = np.array([0.01]*len(I))
aCE = np.sqrt(1 / np.sum(1/dV**2))

print('m,dm,c,dc son:')
print("{:.4f} {:.4f} {:.4f} {:.4f}".format(mReg(I,V), errM(I,V), cReg(I,V), errC(I,V)))
print("common uncertainty =  {:.4f}".format(errCU(I,V)))
print("experimental uncertainty = {:.4f}".format(aCE))

m = mReg(I,V)
c = cReg(I,V)



plt.scatter(I,V)
plt.plot(I,m*I+c)
plt.xlabel("I [\mu A]")
plt.ylabel("V [mV]")
plt.title("V contra I")


residuals = V - (m*I+c)
plt.figure()
plt.plot(I,residuals)
print("Sobre los residuos: (media, std):")
print(np.mean(residuals), np.std(residuals))
















