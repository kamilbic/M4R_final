#############################
# Piecewise Linear Function #
#############################

import numpy as np


def piecewiselinear(ts,pieces):
    rows,cols = ts.shape
    
    plts = np.zeros((rows,(cols-1)*pieces + 1))
    
    for i in range(cols-1):
        cn = ts[:,i]
        cn1 = ts[:,i+1]
        
        diff = (cn1 - cn)/pieces
        for j in range(pieces):
            plts[:,i*pieces + j] = cn + diff*j

    plts[:,-1] = ts[:,-1]
    
    return plts

def collapsepl(plts,pieces):
    rows,cs = plts.shape
    
    cols = int((cs-1)/pieces + 1)
    
    ts = np.zeros((rows,cols))
    
    for i in range(cols-1):
        for j in range(pieces):
            ts[:,i] += plts[:,i*pieces + j]/pieces
    
    ts[:,-1] = plts[:,-1]
    
    return ts
        
def polynomialts(ts,times,pieces,deg):
    pltimes = piecewiselinear(times, pieces)
    
    poly = np.polyfit(times, ts, deg)
    
    polyts = np.polyval(poly,pltimes)
    
    return polyts