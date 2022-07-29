#######################################
# Competing Activator Model Functions #
#######################################

import numpy as np

def camRates(a,b,c,d):
    amaxon = a
    aoff = b
    bmaxon = c
    boff = d
    ksyn = 0
    kdec = 0
    
    return np.array([amaxon,aoff,bmaxon,boff,ksyn,kdec])


def camPropensities(k_rates, x, xt):
    amaxon = k_rates[0]
    aoff = k_rates[1]
    
    bon = k_rates[2]
    bmaxoff = k_rates[3]
    
    ksyn = k_rates[4]
    kdec = k_rates[5]
    
    I = x[0]
    A = x[1]
    B = x[2]
    M = x[3]

    aon = amaxon*xt[0]
    boff = bmaxoff*xt[1]
    
    return np.array([aon*I,aoff*A,bon*I,boff*B,ksyn*A,ksyn*B,kdec*M])

def camNmatrix():
    return np.array([[-1,1,0,0],[1,-1,0,0],[-1,0,1,0],[1,0,-1,0],[0,0,0,1],[0,0,0,1],[0,0,0,-1]])
