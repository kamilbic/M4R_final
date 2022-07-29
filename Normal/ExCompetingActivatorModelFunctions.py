################################################
# Extended Competing Activator Model Functions #
################################################

import numpy as np

def ecamRates(a,b,c,d,e,f):
    amaxon = a
    aoff = b
    bmaxon = c
    boff = d
    cmaxon = e
    coff = f
    ksyn = 0
    kdec = 0
    
    return np.array([amaxon,aoff,bmaxon,boff,cmaxon,coff,ksyn,kdec])


def ecamPropensities(k_rates, x, xt):
    amaxon = k_rates[0]
    aoff = k_rates[1]
    
    bmaxon = k_rates[2]
    boff = k_rates[3]
    
    cmaxon = k_rates[4]
    coff = k_rates[5]
    
    
    ksyn = k_rates[6]
    kdec = k_rates[7]
    
    I = x[0]
    A = x[1]
    B = x[2]
    C = x[3]
    M = x[4]

    aon = amaxon*xt[0]
    bon = bmaxon*xt[1]
    con = cmaxon*xt[2]
    
    return np.array([aon*I,aoff*A,bon*I,boff*B,con*I,coff*C,ksyn*A,ksyn*B,ksyn*C,kdec*M])

def ecamPropensitiesRep1(k_rates, x, xt):
    aon = k_rates[0]
    amaxoff = k_rates[1]
    
    bmaxon = k_rates[2]
    boff = k_rates[3]
    
    cmaxon = k_rates[4]
    coff = k_rates[5]
    
    
    ksyn = k_rates[6]
    kdec = k_rates[7]
    
    I = x[0]
    A = x[1]
    B = x[2]
    C = x[3]
    M = x[4]

    aoff = amaxoff*xt[0]
    bon = bmaxon*xt[1]
    con = cmaxon*xt[2]
    
    return np.array([aon*I,aoff*A,bon*I,boff*B,con*I,coff*C,ksyn*A,ksyn*B,ksyn*C,kdec*M])

def ecamPropensitiesRep2(k_rates, x, xt):
    amaxon = k_rates[0]
    aoff = k_rates[1]
    
    bon = k_rates[2]
    bmaxoff = k_rates[3]
    
    cmaxon = k_rates[4]
    coff = k_rates[5]
    
    
    ksyn = k_rates[6]
    kdec = k_rates[7]
    
    I = x[0]
    A = x[1]
    B = x[2]
    C = x[3]
    M = x[4]

    aon = amaxon*xt[0]
    boff = bmaxoff*xt[1]
    con = cmaxon*xt[2]
    
    return np.array([aon*I,aoff*A,bon*I,boff*B,con*I,coff*C,ksyn*A,ksyn*B,ksyn*C,kdec*M])

def ecamPropensitiesRep3(k_rates, x, xt):
    amaxon = k_rates[0]
    aoff = k_rates[1]
    
    bmaxon = k_rates[2]
    boff = k_rates[3]
    
    con = k_rates[4]
    cmaxoff = k_rates[5]
    
    
    ksyn = k_rates[6]
    kdec = k_rates[7]
    
    I = x[0]
    A = x[1]
    B = x[2]
    C = x[3]
    M = x[4]

    aon = amaxon*xt[0]
    bon = bmaxon*xt[1]
    coff = cmaxoff*xt[2]
    
    return np.array([aon*I,aoff*A,bon*I,boff*B,con*I,coff*C,ksyn*A,ksyn*B,ksyn*C,kdec*M])



def ecamPropensitiesSingleCoop(k_rates, x, xt):
    amaxon = k_rates[0]
    aoff = k_rates[1]
    
    bmaxon = k_rates[2]
    boff = k_rates[3]
    
    cmaxon = k_rates[4]
    coff = k_rates[5]
    
    
    ksyn = k_rates[6]
    kdec = k_rates[7]
    
    I = x[0]
    A = x[1]
    B = x[2]
    C = x[3]
    M = x[4]

    aon = amaxon*xt[0]
    bon = bmaxon*xt[1]
    con = cmaxon*(1 + xt[2]**2)
    
    return np.array([aon*I,aoff*A,bon*I,boff*B,con*I,coff*C,ksyn*A,ksyn*B,ksyn*C,kdec*M])


def ecamPropensitiesSingleCoopRep1(k_rates, x, xt):
    aon = k_rates[0]
    amaxoff = k_rates[1]
    
    bmaxon = k_rates[2]
    boff = k_rates[3]
    
    cmaxon = k_rates[4]
    coff = k_rates[5]
    
    
    ksyn = k_rates[6]
    kdec = k_rates[7]
    
    I = x[0]
    A = x[1]
    B = x[2]
    C = x[3]
    M = x[4]

    aoff = amaxoff*xt[0]
    bon = bmaxon*xt[1]
    con = cmaxon*(1 + xt[2]**2)
    
    return np.array([aon*I,aoff*A,bon*I,boff*B,con*I,coff*C,ksyn*A,ksyn*B,ksyn*C,kdec*M])

def ecamPropensitiesSingleCoopRep2(k_rates, x, xt):
    amaxon = k_rates[0]
    aoff = k_rates[1]
    
    bon = k_rates[2]
    bmaxoff = k_rates[3]
    
    cmaxon = k_rates[4]
    coff = k_rates[5]
    
    
    ksyn = k_rates[6]
    kdec = k_rates[7]
    
    I = x[0]
    A = x[1]
    B = x[2]
    C = x[3]
    M = x[4]

    aon = amaxon*xt[0]
    boff = bmaxoff*xt[1]
    con = cmaxon*(1 + xt[2]**2)
    
    return np.array([aon*I,aoff*A,bon*I,boff*B,con*I,coff*C,ksyn*A,ksyn*B,ksyn*C,kdec*M])

def ecamPropensitiesSingleCoopRep3(k_rates, x, xt):
    amaxon = k_rates[0]
    aoff = k_rates[1]
    
    bmaxon = k_rates[2]
    boff = k_rates[3]
    
    con = k_rates[4]
    cmaxoff = k_rates[5]
    
    
    ksyn = k_rates[6]
    kdec = k_rates[7]
    
    I = x[0]
    A = x[1]
    B = x[2]
    C = x[3]
    M = x[4]

    aon = amaxon*xt[0]
    bon = bmaxon*xt[1]
    coff = cmaxoff*(1 + xt[2]**2)
    
    return np.array([aon*I,aoff*A,bon*I,boff*B,con*I,coff*C,ksyn*A,ksyn*B,ksyn*C,kdec*M])

def ecamPropensitiesDoubleCoop(k_rates, x, xt):
    amaxon = k_rates[0]
    aoff = k_rates[1]
    
    bmaxon = k_rates[2]
    boff = k_rates[3]
    
    cmaxon = k_rates[4]
    coff = k_rates[5]
    
    
    ksyn = k_rates[6]
    kdec = k_rates[7]
    
    I = x[0]
    A = x[1]
    B = x[2]
    C = x[3]
    M = x[4]

    aon = amaxon*xt[0]
    bon = bmaxon*(1 + xt[1]**2)
    con = cmaxon*(1 + xt[2]**2)
    
    return np.array([aon*I,aoff*A,bon*I,boff*B,con*I,coff*C,ksyn*A,ksyn*B,ksyn*C,kdec*M])

def ecamPropensitiesDoubleCoopRep1(k_rates, x, xt):
    aon = k_rates[0]
    amaxoff = k_rates[1]
    
    bmaxon = k_rates[2]
    boff = k_rates[3]
    
    cmaxon = k_rates[4]
    coff = k_rates[5]
    
    
    ksyn = k_rates[6]
    kdec = k_rates[7]
    
    I = x[0]
    A = x[1]
    B = x[2]
    C = x[3]
    M = x[4]

    aoff = amaxoff*xt[0]
    bon = bmaxon*(1 + xt[1]**2)
    con = cmaxon*(1 + xt[2]**2)
    
    return np.array([aon*I,aoff*A,bon*I,boff*B,con*I,coff*C,ksyn*A,ksyn*B,ksyn*C,kdec*M])

def ecamPropensitiesDoubleCoopRep2(k_rates, x, xt):
    amaxon = k_rates[0]
    aoff = k_rates[1]
    
    bon = k_rates[2]
    bmaxoff = k_rates[3]
    
    cmaxon = k_rates[4]
    coff = k_rates[5]
    
    
    ksyn = k_rates[6]
    kdec = k_rates[7]
    
    I = x[0]
    A = x[1]
    B = x[2]
    C = x[3]
    M = x[4]

    aon = amaxon*xt[0]
    boff = bmaxoff*(1 + xt[1]**2)
    con = cmaxon*(1 + xt[2]**2)
    
    return np.array([aon*I,aoff*A,bon*I,boff*B,con*I,coff*C,ksyn*A,ksyn*B,ksyn*C,kdec*M])

def ecamPropensitiesDoubleCoopRep3(k_rates, x, xt):
    amaxon = k_rates[0]
    aoff = k_rates[1]
    
    bmaxon = k_rates[2]
    boff = k_rates[3]
    
    con = k_rates[4]
    cmaxoff = k_rates[5]
    
    
    ksyn = k_rates[6]
    kdec = k_rates[7]
    
    I = x[0]
    A = x[1]
    B = x[2]
    C = x[3]
    M = x[4]

    aon = amaxon*xt[0]
    bon = bmaxon*(1 + xt[1]**2)
    coff = cmaxoff*(1 + xt[2]**2)
    
    return np.array([aon*I,aoff*A,bon*I,boff*B,con*I,coff*C,ksyn*A,ksyn*B,ksyn*C,kdec*M])


def ecamNmatrix():
    return np.array([[-1,1,0,0,0],[1,-1,0,0,0],[-1,0,1,0,0],[1,0,-1,0,0],[-1,0,0,1,0],
                     [1,0,0,-1,0],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,0,-1]])