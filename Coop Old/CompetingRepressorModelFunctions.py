#######################################
# Competing Repressor Model Functions #
#######################################

import numpy as np

def crmRates(a,b,c,d):
    aon1 = a
    aoff1 = b
    aon2 = c
    aoff2 = d
    ksyn = 0
    kdec = 0
    
    return np.array([aon1,aoff1,aon2,aoff2,ksyn,kdec])


def crmPropensities(k_rates, x, xt):
    amaxon1 = k_rates[0]
    aoff1 = k_rates[1]
    
    aon2 = k_rates[2]
    amaxoff2 = k_rates[3]
    
    ksyn = k_rates[4]
    kdec = k_rates[5]
    
    I1 = x[0]
    A = x[1]
    I2 = x[2]
    M = x[3]

    aon1 = amaxon1*xt[0]
    aoff2 = amaxoff2*(1/(1+xt[1]**2))
    return np.array([aon1*I1,aoff1*A,aon2*I2,aoff2*A,ksyn*A,kdec*M])

def crmNmatrix():
    return np.array([[-1,1,0,0],[1,-1,0,0],[0,1,-1,0],[0,-1,1,0],[0,0,0,1],[0,0,0,-1]])
