##########################
# Simple Model Functions #
##########################

import numpy as np


def smRates(a,b):
    kmaxon = a
    koff = b
    ksyn = 0
    kdec = 0
    
    return np.array([kmaxon,koff,ksyn,kdec])

def smRatesHill(a,b,c):
    kmaxon = a
    divider = b
    koff = c
    ksyn = 0
    kdec = 0
    
    return np.array([kmaxon, divider, koff, ksyn, kdec])

def smRatesQuad(a,b,c):
    quad = a
    linear = b
    koff = c
    ksyn = 0
    kdec = 0
    
    return np.array([quad, linear, koff, ksyn, kdec])

def smRatesLogistic(a,b,c,d):
    kmaxon = a
    intercept = b
    lin = c
    koff = d
    ksyn = 0
    kdec = 0
    
    return np.array([kmaxon, intercept, lin, koff, ksyn, kdec])

def smRates110(a,b):
    kmaxon = a
    koff = b
    ksyn = 10
    kdec = 0.01
    
    return np.array([kmaxon,koff,ksyn,kdec])

def smRates011(a,b):
    kmaxon = a
    koff = b
    ksyn = 1
    kdec = 0.01
    
    return np.array([kmaxon,koff,ksyn,kdec])

def smRates00101(a,b):
    kmaxon = a
    koff = b
    ksyn = 0.1
    kdec = 0.01
    
    return np.array([kmaxon,koff,ksyn,kdec])

def smRates10100(a,b):
    kmaxon = a
    koff = b
    ksyn = 100
    kdec = 0.01
    
    return np.array([kmaxon,koff,ksyn,kdec])


def smPropensities(k_rates, x, xt):
    kmaxon = k_rates[0]
    koff = k_rates[1]
    ksyn = k_rates[2]
    kdec = k_rates[3]
    
    I = x[0]
    A = x[1]
    M = x[2]

    kon = kmaxon*xt
    
    return np.array([kon*I,koff*A,ksyn*A,kdec*M])

def smPropensitiesHill(k_rates, x, xt):
    kmaxon = k_rates[0]
    divider = k_rates[1]
    koff = k_rates[2]
    ksyn = k_rates[3]
    kdec = k_rates[4]
    
    I = x[0]
    A = x[1]
    M = x[2]

    kon = kmaxon/(1+(xt/divider))
    
    return np.array([kon*I,koff*A,ksyn*A,kdec*M])

def smPropensitiesQuad(k_rates, x, xt):
    quadratic = k_rates[0]
    linear = k_rates[1]
    koff = k_rates[2]
    ksyn = k_rates[3]
    kdec = k_rates[4]
    
    I = x[0]
    A = x[1]
    M = x[2]

    kon = quadratic*(xt**2) + linear*(xt)
    
    return np.array([kon*I,koff*A,ksyn*A,kdec*M])

def smPropensitiesLogistic(k_rates, x, xt):
    kmaxon = k_rates[0]
    inp = k_rates[1]
    lin = k_rates[2]
    koff = k_rates[3]
    ksyn = k_rates[4]
    kdec = k_rates[5]
    
    
    I = x[0]
    A = x[1]
    M = x[2]

    kon = kmaxon/(1 + np.exp(- inp - lin*xt))
    
    return np.array([kon*I,koff*A,ksyn*A,kdec*M])

def smPropensitiesDouble(k_rates, x, xt):
    kmaxon = k_rates[0]
    kmaxoff = k_rates[1]
    ksyn = k_rates[2]
    kdec = k_rates[3]
    
    I = x[0]
    A = x[1]
    M = x[2]

    kon = kmaxon*xt[0]
    koff = kmaxoff*xt[1]
    
    return np.array([kon*I,koff*A,ksyn*A,kdec*M])

def smPropensitiesDoubleCoop(k_rates, x, xt):
    kmaxon = k_rates[0]
    kmaxoff = k_rates[1]
    ksyn = k_rates[2]
    kdec = k_rates[3]
    
    I = x[0]
    A = x[1]
    M = x[2]

    kon = kmaxon*xt[0]
    koff = kmaxoff*(1/(1+xt[1]**2))
    
    return np.array([kon*I,koff*A,ksyn*A,kdec*M])

def smNmatrix():
    return np.array([[-1,1,0],[1,-1,0],[0,0,1],[0,0,-1]])
    