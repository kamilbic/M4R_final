################################
# Derepression Model Functions #
################################

import numpy as np

def dmRates(a,b,c,d):
    ABrate = a
    BArate = b
    Aon = c
    Aoff = d
    ksyn = 0
    kdec = 0
    
    return np.array([ABrate,BArate,Aon,Aoff,ksyn,kdec])


def dmPropensities(k_rates, x, xt):
    ABrate = k_rates[0]
    BArate = k_rates[1]
    
    Aonrate = k_rates[2]
    Aoff = k_rates[3]
    
    ksyn = k_rates[4]
    kdec = k_rates[5]
    
    I = x[0]
    A = x[1]
    B = x[2]
    M = x[3]

    AB = ABrate*xt[0]
    Aon = Aonrate*xt[1]
    
    return np.array([Aon*I,Aoff*A,AB*A,BArate*B,ksyn*A,ksyn*B,kdec*M])

def dmNmatrix():
    return np.array([[-1,1,0,0],[1,-1,0,0],[0,-1,1,0],[0,1,-1,0],[0,0,0,1],[0,0,0,1],[0,0,0,-1]])