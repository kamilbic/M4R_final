################################################
# Extended Cycling Model Functions #
################################################

import numpy as np

def ecycRates(a,b,c,d):
    amaxon = a
    aoff = b
    bmaxon = c
    boff = d
    ksyn = 0
    kdec = 0
    
    return np.array([amaxon,aoff,bmaxon,boff,ksyn,kdec])


def ecycPropensities(k_rates, x, xt):
    amaxon = k_rates[0]
    aoff = k_rates[1]
    
    bon = k_rates[2]
    bmaxoff = k_rates[3]
    
    ksyn = k_rates[4]
    kdec = k_rates[5]
    
    I1 = x[0]
    A = x[1]
    I2 = x[2]
    B = x[3]
    M = x[4]

    aon = amaxon*xt[0]
    boff = bmaxoff*(1/(1+xt[1]**2))
    
    return np.array([aon*I1,aoff*A,bon*I2,boff*B,ksyn*A,ksyn*B,kdec*M])


def ecycPropensitiesSingleCoop(k_rates, x, xt):
    amaxon = k_rates[0]
    aoff = k_rates[1]
    
    bmaxon = k_rates[2]
    boff = k_rates[3]

    ksyn = k_rates[4]
    kdec = k_rates[5]
    
    I1 = x[0]
    A = x[1]
    I2 = x[2]
    B = x[3]
    M = x[4]

    aon = amaxon*xt[0]
    bon = bmaxon/(1 + xt[1]**2)
    
    return np.array([aon*I1,aoff*A,bon*I2,boff*B,ksyn*A,ksyn*B,kdec*M])


def ecycNmatrix():
    return np.array([[-1,1,0,0,0],[0,-1,1,0,0],[0,0,-1,1,0],[1,0,0,-1,0],
                     [0,0,0,0,1],[0,0,0,0,1],[0,0,0,0,-1]])