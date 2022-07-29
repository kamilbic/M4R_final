############################################
# Active Cycling Model Optimisation - 2 TF #
############################################

from PreCalcExtrande import Extrande
from DataProcess import scaleTS
from DataProcess import scaleTSall
from MIdecodingSimp import estimateMIS
from MIexactEstimate import mi
from ActiveCycleModelFunctions2TF import *
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
    for i in range(m):

        series = [tf1[i,:], tf2[i,:]]
        samTimes = sampleTimes[i,:]
        
        max_t = samTimes[-1]
        times,res,t_ints,state,mrna = Extrande(k_rates,x0,max_t,series,len(series),Propensities, Nmatrix, samTimes)
        resultsA[i,:] = state[1] 
        resultsB[i,:] = state[2]
        results[i,0,:] = state[1]
        results[i,1,:] = state[2]
        statesres += state/m
        resultsColA[i,:] = collapsepl(np.reshape(state[1],(1,-1)), pieces)
        resultsColB[i,:] = collapsepl(np.reshape(state[2],(1,-1)), pieces)
        resultsCol[i,:] = collapsepl(np.reshape(results[i,:],(1,-1)), pieces)
    
    
    
    plt.figure()
    plt.plot(statesres[0],label='I')
    plt.plot(statesres[1],label='A')
    plt.plot(statesres[2],label='B')
    plt.plot(statesres[1]+statesres[2],label='A+B')
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


def MIcomparison(transcriptionFactor1, transcriptionFactor2, method="svm", start=0, pieces=1):
    tf1,origin,sampleTimes = scaleTS(transcriptionFactor1)    
    tf2,origin,sampleTimes = scaleTS(transcriptionFactor2)
    
    n1,_ = tf1.shape
    n2,_ = tf2.shape
    
    sampleTimes = sampleTimes - sampleTimes[0,0]
    
    n = np.min([n1,n2])
    ts = [tf1[:n, :], tf2[:n, :]]
    
    x0 = [1,0,0,0]
    m = n
    testParams = np.load("TestLatticeCycling.npy")
    
    fileName = transcriptionFactor1 + transcriptionFactor2 + "score" + "ActiveCycling" + method
    
    if start == 0:
        testScores = np.zeros((50,1))
    else:
        testScores = np.load(fileName+".npy")
        start = len(testScores[testScores>0])
    
    for i in range(start,50):
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
    
def simMIall(k_rates, ts, x0, m, origin, Propensities, Nmatrix, sTimes, method='svm'):
    tstart = time.time()
    
    resultsA = [np.zeros((m,len(ts[i][0][0,:]))) for i in range(len(ts))]
    resultsB = [np.zeros((m,len(ts[i][0][0,:]))) for i in range(len(ts))]
    
    stress_results = [0]*len(ts)
    rich_results = [0]*len(ts)
     
    k_rates = activeCycleRates(k_rates[0],k_rates[1],k_rates[2])
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
    
def MIcomparisonAll(transcriptionFactor1, transcriptionFactor2, method="svm", start=0, pieces=1):
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
        
    x0 = [1,0,0,0]
    m = n
    
    testParams = np.load("TestLattice.npy")
    
    fileName = transcriptionFactor1 + transcriptionFactor2 + "all" + "ActiveCycling" + method
    
    if start == 0:
        testScores = np.zeros((40,1))
    else:
        testScores = np.load(fileName+".npy")
        start = len(testScores[testScores>0])
    
    for i in range(start,10):
        params = testParams[i]
        a = params[0]
        b = params[1]
        c = params[2]
        print(params)
        k_rates = activeCycleRates(a,b,c)
        testScores[i] = simMIall(k_rates, ts, x0, m, origin, activeCyclePropensities, activeCycleNmatrix, sampleTimes, method)
        print(testScores[i])
        #np.save(fileName,testScores)
        
    print(testScores)
    #np.save(fileName,testScores)
    
def particleSwarmStartACall(transcriptionFactor1, transcriptionFactor2, progress, pieces=1):
    tf1,origin,sampleTimes = scaleTSall(transcriptionFactor1)    
    tf2,origin,sampleTimes = scaleTSall(transcriptionFactor2)
    
    n1,_ = tf1[0].shape
    n2,_ = tf2[0].shape
    
    n = np.min([n1,n2])
    ts1 = [tf1[0][:n, :], tf2[0][:n, :]]
    ts2 = [tf1[1][:n, :], tf2[1][:n, :]]
    ts3 = [tf1[2][:n, :], tf2[2][:n, :]]
    ts = [ts1, ts2, ts3]
    x0 = [1,0,0,0]
    m = np.min([n,100])
    print(m)
    
    for i in range(len(sampleTimes)):
        sampleTimes[i] = sampleTimes[i] - sampleTimes[i][0,0]

    
    lattice = np.load("latticeActiveCycling.npy")
    
    startParticleSwarm('ActiveCyclingMixed', lattice, [[1,800],[1,800],[1,800]],
                       10000, simMIall, 8, ts, x0, m, origin, activeCyclePropensities, activeCycleNmatrix,
                       sampleTimes, pieces, progress)

def particleSwarmContinueACall(transcriptionFactor1, transcriptionFactor2,maxits, pieces=1):
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
    
    x0 = [1,0,0,0]
    m = np.min([n,100])
    print(m)
    
    progress = np.load("progressActiveCyclingMixed.npy")
    lattice = np.load("latticeActiveCycling.npy")
    bestPos = np.load("bestPosActiveCyclingMixed.npy")
    bestVals = np.load("bestValsActiveCyclingMixed.npy")
    swarmPos = np.load("swarmPosActiveCyclingMixed.npy")
    swarmVal = 1*np.load("swarmValsActiveCyclingMixed.npy")
    currentPos = np.load("currentPosActiveCyclingMixed.npy")
    currentVals = np.load("currentValsActiveCyclingMixed.npy")
    velocities = np.load("velocitiesActiveCyclingMixed.npy")
    parDict = json.load(open("parDictActiveCyclingMixed.json"))
    
    modelName = 'ActiveCyclingMixed'
    
    continueParticleSwarm(lattice, parDict, currentPos, currentVals, bestPos,
                              bestVals, swarmPos, swarmVal, velocities, progress,
                              modelName, simMIall, maxits, ts, x0, m, origin, 
                              activeCyclePropensities, activeCycleNmatrix, sampleTimes, pieces)
    
def fn():
    particleSwarmStartACall("dot","mig1",np.zeros(10))
    particleSwarmContinueACall("dot","mig1",8)