#################################################
# Competing Activator Repressor Model Functions #
#################################################

import numpy as np

def carmRates(a,b,c,d):
    amaxon = a
    aoff = b
    to12 = c
    to21 = d
    ksyn = 0
    kdec = 0
    
    return np.array([amaxon,aoff,to12,to21,ksyn,kdec])


def carmPropensities(k_rates, x, xt):
    amaxon = k_rates[0]
    aoff = k_rates[1]
    
    to12 = k_rates[2]
    to21 = k_rates[3]
    
    ksyn = k_rates[4]
    kdec = k_rates[5]
    
    I1 = x[0]
    A = x[1]
    I2 = x[2]
    M = x[3]

    aon = amaxon*xt[0]
    to12 = to12*(xt[1]**2)/(1 + xt[1]**2)
    
    return np.array([aon*I1,aoff*A,to12*I1,to21*I2,ksyn*A,kdec*M])

def carmNmatrix():
    return np.array([[-1,1,0,0],[1,-1,0,0],[-1,0,1,0],[1,0,-1,0],[0,0,0,1],[0,0,0,-1]])