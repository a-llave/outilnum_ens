"""
Description:    wiener filter

Author: Martin Witt - ENS Rennes
Date: 30/01/2020

Version: 1

Date    | Auth. | Vers.  |  Comments
30/01/20   MWi     1.0       Initialization

"""

import numpy as np
import sys


class Wiener(F,N,x_v,s_v,n_v):
    def __init__(F,N):
        # PARAMETERS
        self.N = N
        self.s_m = zeros(F,N)
        
    def process(x_v,s_v,n_v):
        #estimation
        self.x_m = [self.x_m[:,:1],x_v]
        self.s_m = [self.s_m[:,:1],s_v]
        self.n_m = [self.n_m[:,:1],n_v]
        
        sigma_s2 = mean(abs(self.s_m)**2)
        sigma_n2 = mean(abs(self.n_m)**2)
        
        #filter
        w = sigma_s2/(sigma_n2+sigma_s2)
        return w*x_m

        