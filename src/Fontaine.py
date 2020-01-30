# encoding: utf-8
"""
Description: Class Fontaine for ENS mini project

Author: Victor Perrin - ENS Rennes
Date: 30/01/2020

Version: 1.0

Date    | Auth. | Vers.  |  Comments
30/01/20  VPe     1.0       Initialization



"""

import numpy as np
import sys
import scipy.stats as scistat

class Fontaine:
    def __init__(alpha,N,l,f,gamma):
        self.N=N
        self.S=zeros(F,N)
        self.alpha=alpha
        self.l=l
        self.f=f
        self.gamma=0.5772156 #contante d'Euler
        
    def estimation(s):
        for i in range(l-self.N,self.l+1):
            sigmac2_s+=abs(s(i,self.f))**2
        return (1/self.N)*sigmac2_s
    
    def process(x,s,n):
        self.S=[self.s(),s]
        #estimation
        lambdac_s=np.exp((1/self.alpha)*estimation(ln(abs(self.s)))+self.gamma*(1-1/self.alpha))
        phic=scistat.levy.median(loc=0,scale=1) #diffère suivant les valeurs de alpha
        #filtrage
        w=(phic*lambdac_s**2)/(phic*lambdac_s**2+phic*lambdan**2)
        sc=numpy.convolve(w,x)
        return sc