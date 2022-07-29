#######################################
# Inactive Cycling Model Optimisation #
#######################################

from PreCalcExtrande import Extrande
from DataProcess import scaleTS
from DataProcess import scaleTSall
from MIdecodingSimp import estimateMIS
from MIexactEstimate import mi
from ActiveCycleModelFunctions1TF import *
from ParticleSwarmContinual import startParticleSwarm
from ParticleSwarmContinual import continueParticleSwarm
from PiecewiseLinear import piecewiselinear
from PiecewiseLinear import collapsepl
import matplotlib.pyplot as plt
import numpy as np
import time
import json

def simMI(k_rates, ts, x0, m, origin, Propensities, Nmatrix, sampleTimes, pieces, method):
    tstart = time.time()
    origin = origin - 1
    sm0 = sampleTimes[0,0]
    sampleTimes = sampleTimes - sm0
    sampleTimes = piecewiselinear(sampleTimes, pieces)
    
    resultsCol = np.zeros((m,len(ts[0,:])))

    ts = piecewiselinear(ts, pieces)

    results = np.zeros(sampleTimes.shape)
    
    statesres = 0  

    for i in range(m):
        series = ts[i,:]
        samTimes = sampleTimes[i,:]
        
        max_t = samTimes[-1]
        times,res,t_ints,state, RNA_state = Extrande(k_rates, x0, max_t, series,
                                                     len(series), Propensities, 
                                                     Nmatrix, samTimes)
        results[i,:] = state[1] + state[2]
        
        #state = np.reshape(state[1], (1,-1))
        statesres += state/m
        resultsCol[i,:] = collapsepl((state[1]+state[2]).reshape((1,-1)), pieces)
     
    rich_results = results[:,origin*pieces-20*pieces:origin*pieces]
    stress_results = results[:,origin*pieces:origin*pieces+20*pieces]
    
    plt.figure()
    plt.plot(statesres[0],label='I')
    plt.plot(statesres[1],label='A')
    plt.plot(statesres[2],label='B')
    plt.plot(statesres[1]+statesres[2],label='A+B')
    plt.legend(loc='upper left')
    plt.show()

    if method == "svm":
        MI = estimateMIS([[stress_results, rich_results]])
        ttaken = time.time() - tstart
        print("Time Taken: ", ttaken)
    
        return MI[0]
    else:
        inp = np.r_[stress_results, rich_results]
        out = np.r_[np.zeros(rich_results.shape[0]),np.ones(stress_results.shape[0])]
        MI = mi(inp,out)
        ttaken = time.time() - tstart
        print("Time Taken: ", ttaken)
    
        return MI

def simpleMI(params, ts, x0, m, origin, Propensities, Nmatrix, sampleTimes, pieces=1):
    a = params[0]
    b = params[1]
    c = params[2]
    krates = activeCycleRates(a,b,c)
    MI = simMI(krates, ts, x0, m, origin, Propensities, Nmatrix, sampleTimes, pieces)
    return MI

def MIcomparison(transcriptionFactor, method="svm", start=0, pieces=1):
    tf,origin,sampleTimes = scaleTS(transcriptionFactor)    
    
    n,_ = tf.shape
    
    sampleTimes = sampleTimes - sampleTimes[0,0]
    
    ts = tf
    
    x0 = [1,0,0,0]
    m = n
    
    testParams = np.load("TestLatticeCycling.npy")
    
    fileName = transcriptionFactor + "score" + "ActiveCycling" + method
    
    if start == 0:
        testScores = np.zeros((50,1))
    else:
        testScores = np.load(fileName+".npy")
        start = len(testScores[testScores>0])
    
    for i in range(start,50):
        print(i+1)
        params = testParams[i]
        a = params[0]
        b = params[1]
        c = params[2]
        print(params)
        k_rates = activeCycleRates(a,b,c)
        testScores[i] = simMI(k_rates, ts, x0, m, origin, 
                  activeCyclePropensities, activeCycleNmatrix, sampleTimes, 
                  pieces, method)
        print(testScores[i])
        np.save(fileName,testScores)
        
    print(testScores)
    np.save(fileName,testScores)
    
def optimalB(transcriptionFactor,kon, method="svm", pieces=1):
    blow = np.linspace(kon/3,kon,6)
    bhigh = np.linspace(kon,3*kon,6)
    
    bs = np.r_[blow[:-1],bhigh]
    scores = np.zeros(bs.shape)
    
    tf,origin,sampleTimes = scaleTS(transcriptionFactor)    
    
    n,_ = tf.shape
    
    sampleTimes = sampleTimes - sampleTimes[0,0]
    
    ts = tf
    
    x0 = [1,0,0,0]
    m = n
    
    for i in range(len(bs)):
        b = bs[i]
        krates = activeCycleRates(kon,b,b)
        scores[i] = simMI(krates, ts, x0, m, origin, 
                  activeCyclePropensities, activeCycleNmatrix, sampleTimes, 
                  pieces, method)
        print(scores[i], b)
        
    maxb = bs[np.argmax(scores)]
    maxmi = np.max(scores)
    return maxb, maxmi
        
def optimalBs(tF,l,h,k):
    kons = 10**(np.linspace(l,h,k))
    koffs = np.zeros(kons.shape)
    scores = np.zeros(kons.shape)

    for i in range(len(kons)):
        kon = kons[i]
        print(kon)
        koff, maxmi = optimalB(tF,kon)
        koffs[i] = koff
        scores[i] = maxmi
        
    return koffs, scores

def simMIall(k_rates, ts, x0, m, origin, Propensities, Nmatrix, sTimes, method):
    tstart = time.time()
    for i in range(len(ts)):
        origin[i] = origin[i] - 1
    
    resultsA = [np.zeros((m,len(ts[i][0,:]))) for i in range(len(ts))]
    resultsB = [np.zeros((m,len(ts[i][0,:]))) for i in range(len(ts))]
    
    stress_results = [0]*len(ts)
    rich_results = [0]*len(ts)
     
    k_rates = activeCycleRates(k_rates[0],k_rates[1],k_rates[2])
    for j in range(len(ts)):
        tf1 = ts[j]
        sampleTimes = sTimes[j]
        sm0 = sampleTimes[0,0]
        sampleTimes = sampleTimes - sm0
        for i in range(m):
            series = tf1[i,:]
            samTimes = sampleTimes[i,:]
            
            max_t = samTimes[-1]
            times,res,t_ints,state,mrna = Extrande(k_rates,x0,max_t,series,len(series),Propensities, Nmatrix, samTimes)
            resultsA[j][i,:] = state[1] 
            resultsB[j][i,:] = state[2]
        
        orig = origin[j]
        stress_results[j] = resultsA[j][:,orig:orig+20] + resultsB[j][:,orig:orig+20]
        rich_results[j] = resultsA[j][:,orig-20:orig] + resultsB[j][:,orig-20:orig]
        
    
    if method == "svm":
        reslist = [rich_results[0]]
        for i in range(len(ts)):
            reslist.append(stress_results[i])
        
        MI = estimateMIS([reslist])
        ttaken = time.time() - tstart
        print("Time Taken: ", ttaken)
    
        return MI[0]
    
    else:
        inp = rich_results[0]
        out = np.zeros(rich_results[0].shape[0])
        for i in range(len(ts)):
            inp = np.r_[inp,stress_results[i]]
            out = np.r_[out,(i+1)*np.ones(stress_results[i].shape[0])]

        MI = mi(inp,out)
        ttaken = time.time() - tstart
        print("Time Taken: ", ttaken)
    
        return MI

def optimalBall(transcriptionFactor,kon, method="svm", pieces=1):
    blow = np.linspace(kon/3,kon,6)
    bhigh = np.linspace(kon,3*kon,6)
    
    bs = np.r_[blow[:-1],bhigh]
    scores = np.zeros(bs.shape)
    
    tf,origin,sampleTimes = scaleTSall(transcriptionFactor)    
    
    n,_ = tf[0].shape
    
    ts1 = tf[0][:n, :]
    ts2 = tf[1][:n, :]
    ts3 = tf[2][:n, :]
    ts = [ts1, ts2, ts3]
    x0 = [1,0,0,0]
    m = n

    for i in range(len(sampleTimes)):
        sampleTimes[i] = sampleTimes[i] - sampleTimes[i][0,0]

    
    for i in range(len(bs)):
        b = bs[i]
        krates = activeCycleRates(kon,b,b)
        scores[i] = simMIall(krates, ts, x0, m, origin, 
                  activeCyclePropensities, activeCycleNmatrix, sampleTimes, 
                  method)
        print(scores[i], b)
        
    maxb = bs[np.argmax(scores)]
    maxmi = np.max(scores)
    return maxb, maxmi

def optimalBsall(tF,l,h,k):
    kons = 10**(np.linspace(l,h,k))
    koffs = np.zeros(kons.shape)
    scores = np.zeros(kons.shape)
    fileName = tF +"ActAll"+ str(l)+"and"+str(h)
    print(fileName)
    for i in range(len(kons)):
        kon = kons[i]
        print(kon)
        koff, maxmi = optimalBall(tF,kon)
        koffs[i] = koff
        scores[i] = maxmi
    
        scoreArray = np.array([koffs,scores])
    
        np.save(fileName,scoreArray)
        
    return koffs, scores