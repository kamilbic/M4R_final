#######################################
# Active Cycle Model Functions - 2 TF #
#######################################

import numpy as np

def activeCycleRates(a,b,c):
    kIA = a
    kAB = b
    kBI = c
    
    return np.array([kIA, kAB, kBI])


def activeCyclePropensities(k_rates, x, xt):
    kIA = k_rates[0]
    kAB = k_rates[1]
    kBI = k_rates[2]
    
    I = x[0]
    A = x[1]
    B = x[2]
    M = x[3]
    
    kon = kIA*xt[0]
    kswitch = kAB
    koff = kBI*(1/(1+xt[1]**2))
    
    return np.array([kon*I,kswitch*A,koff*B])

def activeCycleNmatrix():
    return np.array([[-1,1,0,0],[0,-1,1,0],[1,0,-1,0]])