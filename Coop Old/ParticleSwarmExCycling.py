from PreCalcExtrande import Extrande
from DataProcess import scaleTS
from DataProcess import scaleTSall
from MIdecodingSimp import estimateMIS
from MIexactEstimate import mi
from ExCyclingModelFunctions import *
from ParticleSwarmContinual import startParticleSwarm
from ParticleSwarmContinual import continueParticleSwarm
from PiecewiseLinear import piecewiselinear
from PiecewiseLinear import collapsepl
import numpy as np
import time
import json
from matplotlib import pyplot as plt


def simMI(k_rates, ts, x0, m, origin, Propensities, Nmatrix, sampleTimes, pieces, method):
    tstart = time.time()
    origin = origin - 1
    sm0 = sampleTimes[0,0]
    sampleTimes = sampleTimes - sm0
    sampleTimes = piecewiselinear(sampleTimes, pieces)
    
    resultsColA = np.zeros((m,len(ts[0][0,:])))
    resultsColB = np.zeros((m,len(ts[0][0,:])))
    
    resultsCol = np.zeros((m,2*len(ts[0][0,:])))
    
    tf1 = ts[0]
    tf2 = ts[1]
    
    tf1 = piecewiselinear(tf1, pieces)
    tf2 = piecewiselinear(tf2, pieces)

    
    resultsA = np.zeros((m,len(ts[0][0,:])))
    resultsB = np.zeros((m,len(ts[0][0,:])))
    
    results = np.zeros((m,2,len(ts[0][0,:])))
    
    statesres = 0       
    k_rates = ecycRates(k_rates[0],k_rates[1],k_rates[2],k_rates[3])
    for i in range(m):

        series = [tf1[i,:], tf2[i,:]]
        samTimes = sampleTimes[i,:]
        
        max_t = samTimes[-1]
        times,res,t_ints,state,mrna = Extrande(k_rates,x0,max_t,series,len(series),Propensities, Nmatrix, samTimes)
        resultsA[i,:] = state[1] 
        resultsB[i,:] = state[3]
        results[i,0,:] = state[1]
        results[i,1,:] = state[3]
        statesres += state/m
        resultsColA[i,:] = collapsepl(np.reshape(state[1],(1,-1)), pieces)
        resultsColB[i,:] = collapsepl(np.reshape(state[3],(1,-1)), pieces)
        resultsCol[i,:] = collapsepl(np.reshape(results[i,:],(1,-1)), pieces)
    
    
    
    plt.figure()
    plt.plot(statesres[0]+statesres[2],label='I')
    plt.plot(statesres[1],label='A')
    plt.plot(statesres[3],label='B')
    plt.plot(statesres[1]+statesres[3],label='A+B')
    plt.legend(loc='upper left')
    plt.show()
    
    stress_results = results[:,:,origin:origin+20]
    rich_results = results[:,:,origin-20:origin]
    
    stress_results_A = stress_results[:,0]
    rich_results_A = rich_results[:,0]
    stress_results_B = stress_results[:,1]
    rich_results_B = rich_results[:,1]
    
    if method == "svm":
        MI = estimateMIS([[stress_results_A + stress_results_B, rich_results_A + rich_results_B]])
        ttaken = time.time() - tstart
        print("Time Taken: ", ttaken)
    
        return MI[0]
    else:
        inp = np.r_[rich_results_A + rich_results_B, stress_results_A + stress_results_B]
        out = np.r_[np.zeros(rich_results.shape[0]),np.ones(stress_results.shape[0])]
        MI = mi(inp,out)
        ttaken = time.time() - tstart
        print("Time Taken: ", ttaken)
    
        return MI

def optimalB(transcriptionFactor,kon, method="svm", pieces=1):
    blow = np.linspace(kon/3,kon,6)
    bhigh = np.linspace(kon,3*kon,6)
    
    bs = np.r_[blow[:-1],bhigh]
    scores = np.zeros(bs.shape)
    
    tf,origin,sampleTimes = scaleTS(transcriptionFactor)    
    ts = [tf,tf]
    n,_ = tf.shape
    
    sampleTimes = sampleTimes - sampleTimes[0,0]

    
    x0 = [1,0,0,0,0]
    m = n
    
    for i in range(len(bs)):
        b = bs[i]
        krates = ecycRates(kon,b,kon,b)
        scores[i] = simMI(krates, ts, x0, m, origin, 
                  ecycPropensities, ecycNmatrix, sampleTimes, 
                  pieces, method)
        print(kon,scores[i], b)
        
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



def simMIall(k_rates, ts, x0, m, origin, Propensities, Nmatrix, sTimes, method='svm'):
    tstart = time.time()
    
    resultsA = [np.zeros((m,len(ts[i][0][0,:]))) for i in range(len(ts))]
    resultsB = [np.zeros((m,len(ts[i][0][0,:]))) for i in range(len(ts))]
    
    stress_results = [0]*len(ts)
    rich_results = [0]*len(ts)
     
    k_rates = ecycRates(k_rates[0],k_rates[1],k_rates[2],k_rates[3])
    for j in range(len(ts)):
        tf1 = ts[j][0]
        tf2 = ts[j][1]
        sampleTimes = sTimes[j]
        sm0 = sampleTimes[0,0]
        sampleTimes = sampleTimes - sm0
        for i in range(m):
            series = [tf1[i,:], tf2[i,:]]
            samTimes = sampleTimes[i,:]
            
            max_t = samTimes[-1]
            times,res,t_ints,state,mrna = Extrande(k_rates,x0,max_t,series,len(series),Propensities, Nmatrix, samTimes)
            resultsA[j][i,:] = state[1] 
            resultsB[j][i,:] = state[3]
        
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
    ts1 = [tf[0][:n, :], tf[0][:n, :]]
    ts2 = [tf[1][:n, :], tf[1][:n, :]]
    ts3 = [tf[2][:n, :], tf[2][:n, :]]
    ts = [ts1, ts2, ts3]
    
    for i in range(len(sampleTimes)):
        sampleTimes[i] = sampleTimes[i] - sampleTimes[i][0,0]

    
    x0 = [1,0,0,0,0]
    m = n
    
    for i in range(len(bs)):
        b = bs[i]
        krates = ecycRates(kon,b,kon,b)
        scores[i] = simMIall(krates, ts, x0, m, origin, 
                  ecycPropensities, ecycNmatrix, sampleTimes, 
                  method)
        print(kon,scores[i], b)
        
    maxb = bs[np.argmax(scores)]
    maxmi = np.max(scores)
    return maxb, maxmi
        
def optimalBsall(tF,l,h,k):
    kons = 10**(np.linspace(l,h,k))
    koffs = np.zeros(kons.shape)
    scores = np.zeros(kons.shape)
    fileName = tF +"ExcycAll"+ str(l)+"and"+str(h)
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

def particleSwarmStartExCall(transcriptionFactor1, transcriptionFactor2, progress, pieces=1):
    tf1,origin,sampleTimes = scaleTSall(transcriptionFactor1)    
    tf2,origin,sampleTimes = scaleTSall(transcriptionFactor2)
    
    n1,_ = tf1[0].shape
    n2,_ = tf2[0].shape
    
    n = np.min([n1,n2])
    ts1 = [tf1[0][:n, :], tf2[0][:n, :]]
    ts2 = [tf1[1][:n, :], tf2[1][:n, :]]
    ts3 = [tf1[2][:n, :], tf2[2][:n, :]]
    ts = [ts1, ts2, ts3]
    x0 = [1,0,0,0,0]
    m = np.min([n,100])
    
    for i in range(len(ts)):
        origin[i] = origin[i] - 1
    print(m)
    
    for i in range(len(sampleTimes)):
        sampleTimes[i] = sampleTimes[i] - sampleTimes[i][0,0]

    
    lattice = np.load("latticeCompetingActivator.npy")
    
    startParticleSwarm('ExCyclingMixed', lattice, [[1,800],[1,800],[1,800],[1,800]],
                       10000, simMIall, 10, ts, x0, m, origin, ecycPropensities, ecycNmatrix,
                       sampleTimes, pieces, progress)

def particleSwarmContinueExCall(transcriptionFactor1, transcriptionFactor2,maxits, pieces=1):
    tf1,origin,sampleTimes = scaleTSall(transcriptionFactor1)    
    tf2,origin,sampleTimes = scaleTSall(transcriptionFactor2)
    
    n1,_ = tf1[0].shape
    n2,_ = tf2[0].shape
    
    n = np.min([n1,n2])
    ts1 = [tf1[0][:n, :], tf2[0][:n, :]]
    ts2 = [tf1[1][:n, :], tf2[1][:n, :]]
    ts3 = [tf1[2][:n, :], tf2[2][:n, :]]
    ts = [ts1, ts2, ts3]
    
    for i in range(len(ts)):
        origin[i] = origin[i] - 1
        
    x0 = [1,0,0,0,0]
    m = n
    print(m)
    
    progress = np.load("progressExCyclingMixed.npy")
    lattice = np.load("latticeExCyclingMixed.npy")
    bestPos = np.load("bestPosExCyclingMixed.npy")
    bestVals = np.load("bestValsExCyclingMixed.npy")
    swarmPos = np.load("swarmPosExCyclingMixed.npy")
    swarmVal = 1*np.load("swarmValsExCyclingMixed.npy")
    currentPos = np.load("currentPosExCyclingMixed.npy")
    currentVals = np.load("currentValsExCyclingMixed.npy")
    velocities = np.load("velocitiesExCyclingMixed.npy")
    parDict = json.load(open("parDictExCyclingMixed.json"))
    
    modelName = 'ExCyclingMixed'
    
    continueParticleSwarm(lattice, parDict, currentPos, currentVals, bestPos,
                              bestVals, swarmPos, swarmVal, velocities, progress,
                              modelName, simMIall, maxits, ts, x0, m, origin, 
                              ecycPropensities, ecycNmatrix, sampleTimes, pieces)
    
def fn():
    particleSwarmStartExCall("dot","mig1",np.zeros(10))
    particleSwarmContinueExCall("dot","mig1",8)