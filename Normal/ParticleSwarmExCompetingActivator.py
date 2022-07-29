from PreCalcExtrande import Extrande
from DataProcess import scaleTS
from DataProcess import scaleTSall
from MIdecodingSimp import estimateMIS
from MIexactEstimate import mi
from ExCompetingActivatorModelFunctions import *
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
        
    tf1 = ts[0]
    tf2 = ts[1]
    tf3 = ts[2]
    
    tf1 = piecewiselinear(tf1, pieces)
    tf2 = piecewiselinear(tf2, pieces)
    tf3 = piecewiselinear(tf3, pieces)
    
    resultsA = np.zeros((m,len(ts[0][0,:])))
    resultsB = np.zeros((m,len(ts[0][0,:])))
    resultsC = np.zeros((m,len(ts[0][0,:])))
    
    results = np.zeros((m,3,len(ts[0][0,:])))
    
    statesres = 0       
    for i in range(m):
        series = [tf1[i,:], tf2[i,:], tf3[i,:]]
        samTimes = sampleTimes[i,:]
        
        max_t = samTimes[-1]
        times,res,t_ints,state,mrna = Extrande(k_rates,x0,max_t,series,len(series),Propensities, Nmatrix, samTimes)
        resultsA[i,:] = state[1] 
        resultsB[i,:] = state[2]
        resultsC[i,:] = state[3]
        results[i,0,:] = state[1]
        results[i,1,:] = state[2]
        results[i,2,:] = state[3]
        statesres += state/m
    
    plt.figure()
    plt.plot(statesres[0],label='I')
    plt.plot(statesres[1],label='A')
    plt.plot(statesres[2],label='B')
    plt.plot(statesres[3],label='C')
    plt.plot(statesres[1]+statesres[2]+statesres[3],label='A+B+C')
    plt.legend(loc='upper left')
    plt.show()
    
    stress_results = results[:,:,origin:origin+20]
    rich_results = results[:,:,origin-20:origin]
    
    if method == "svm":
        MI = estimateMIS([[stress_results.sum(axis=1), rich_results.sum(axis=1)]])
        ttaken = time.time() - tstart
        print("Time Taken: ", ttaken)
    
        return MI[0]
    
    else:
        inp = np.r_[rich_results.sum(axis=1), stress_results.sum(axis=1)]
        out = np.r_[np.zeros(rich_results.shape[0]),np.ones(stress_results.shape[0])]
        MI = mi(inp,out)
        ttaken = time.time() - tstart
        print("Time Taken: ", ttaken)
    
        return MI
    

def MIcomparison(tFac1, tFac2, tFac3, coop, method="svm", start=0, pieces=1):
    tf1,origin,sampleTimes = scaleTS(tFac1)    
    tf2,origin,sampleTimes = scaleTS(tFac2)
    tf3,origin,sampleTimes = scaleTS(tFac3)
    
    n1,_ = tf1.shape
    n2,_ = tf2.shape
    n3,_ = tf3.shape
    
    sampleTimes = sampleTimes - sampleTimes[0,0]
    
    n = np.min([n1,n2,n3])
    ts = [tf1[:n], tf2[:n], tf3[:n]]
    
    x0 = [1,0,0,0,0]
    m = n
    
    testParams = np.load("TestLattice4State.npy")
    
    fileName = tFac1 + tFac2 + tFac3 + "All" + "CompetingActivator" + method + "coop" + str(coop)
    
    if start == 0:
        testScores = np.zeros((50,1))
    else:
        testScores = np.load(fileName+".npy")
        start = len(testScores[testScores>0])
        
    if coop == 0:
        if tFac1 == "maf1":
            prop = ecamPropensitiesRep1
        if tFac2 == "maf1":
            prop = ecamPropensitiesRep2
        if tFac3 == "maf1":
            prop = ecamPropensitiesRep3
        else:
            prop = ecamPropensities
    elif coop == 1:
        if tFac1 == "maf1":
            prop = ecamPropensitiesSingleCoopRep1
        if tFac2 == "maf1":
            prop = ecamPropensitiesSingleCoopRep2
        if tFac3 == "maf1":
            prop = ecamPropensitiesSingleCoopRep3
        else:
            prop = ecamPropensitiesSingleCoop
    else:
        if tFac1 == "maf1":
            prop = ecamPropensitiesDoubleCoopRep1
        if tFac2 == "maf1":
            prop = ecamPropensitiesDoubleCoopRep2
        if tFac3 == "maf1":
            prop = ecamPropensitiesDoubleCoopRep3
        else:
            prop = ecamPropensitiesDoubleCoop
            
            
    for i in range(start,50):
        params = testParams[i]
        a = params[0]
        b = params[1]
        c = params[2]
        d = params[3]
        e = params[4]
        f = params[5]
        print(params)
        k_rates = ecamRates(a,b,c,d,e,f)
        testScores[i] = simMI(k_rates, ts, x0, m, origin, prop, ecamNmatrix, sampleTimes, pieces, method)
        print(testScores[i])
        np.save(fileName,testScores)
        
    print(testScores)
    np.save(fileName,testScores)
    
    
def simMIall(k_rates, ts, x0, m, origin, Propensities, Nmatrix, sTimes, method):
    tstart = time.time()
    
    resultsA = [np.zeros((m,len(ts[i][0][0,:]))) for i in range(len(ts))]
    resultsB = [np.zeros((m,len(ts[i][0][0,:]))) for i in range(len(ts))]
    
    stress_results = [0]*len(ts)
    rich_results = [0]*len(ts)
     
    k_rates = ecamRates(k_rates[0],k_rates[1],k_rates[2],k_rates[3],k_rates[4],k_rates[5])
    for j in range(len(ts)):
        tf1 = ts[j][0]
        tf2 = ts[j][1]
        tf3 = ts[j][2]
        sampleTimes = sTimes[j]
        sm0 = sampleTimes[0,0]
        sampleTimes = sampleTimes - sm0
        for i in range(m):
            series = [tf1[i,:], tf2[i,:], tf3[i,:]]
            samTimes = sampleTimes[i,:]
            max_t = samTimes[-1]
            times,res,t_ints,state,mrna = Extrande(k_rates,x0,max_t,series,len(series),Propensities, Nmatrix, samTimes)
            resultsA[j][i,:] = state[1] 
            resultsB[j][i,:] = state[2]
        
        orig = origin[j]
        stress_results[j] = resultsA[j][:,orig:orig+20] + resultsB[j][:,orig:orig+20]
        rich_results[j] = resultsA[j][:,orig-20:orig] + resultsB[j][:,orig-20:orig]
        
    
    plt.figure()
    plt.plot(rich_results[0].mean(axis=0),label='rich')
    plt.plot(stress_results[0].mean(axis=0), label='stress1')
    plt.plot(stress_results[1].mean(axis=0), label='stress2')
    plt.plot(stress_results[2].mean(axis=0), label='stress3')
    plt.legend(loc='upper left')
    plt.show()
    
    #if method == "svm":
    reslist = [rich_results[0]]
    for i in range(len(ts)):
        reslist.append(stress_results[i])
    
    MI = estimateMIS([reslist])
    ttaken = time.time() - tstart
    print("Time Taken: ", ttaken)

    return MI[0]
    '''
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
    '''
    
def MIcomparisonAll(tFac1, tFac2, tFac3, coop, method="svm", start=0, pieces=1):
    tf1,origin,sampleTimes = scaleTSall(tFac1)    
    tf2,origin,sampleTimes = scaleTSall(tFac2)
    tf3,origin,sampleTimes = scaleTSall(tFac3)
    
    n1,_ = tf1[0].shape
    n2,_ = tf2[0].shape
    n3,_ = tf3[0].shape
    
    n = np.min([n1,n2,n3])
    ts1 = [tf1[0][:n, :], tf2[0][:n, :], tf3[0][:n, :]]
    ts2 = [tf1[1][:n, :], tf2[1][:n, :], tf3[1][:n, :]]
    ts3 = [tf1[2][:n, :], tf2[2][:n, :], tf3[2][:n, :]]
    ts = [ts1, ts2, ts3]
    
    x0 = [1,0,0,0,0]
    m = n
    
    testParams = np.load("TestLattice4State.npy")
    
    fileName = tFac1 + tFac2 + tFac3 + "score" + "CompetingActivator" + method + "coop" + str(coop)
    
    if start == 0:
        testScores = np.zeros((50,1))
        np.save(fileName,testScores)
    else:
        if start < 25:
            testScores = np.load(fileName+".npy")
            start = sum(testScores[:25]>0)[0]
            end = 25
        else:
            testScores = np.load(fileName+".npy")
            start = sum(testScores[25:]>0)[0] + 25
            end = 50
        #start = len(testScores[testScores>0])
        
    if coop == 0:
        if tFac1 == "maf1":
            prop = ecamPropensitiesRep1
        if tFac2 == "maf1":
            prop = ecamPropensitiesRep2
        if tFac3 == "maf1":
            prop = ecamPropensitiesRep3
        else:
            prop = ecamPropensities
    elif coop == 1:
        if tFac1 == "maf1":
            prop = ecamPropensitiesSingleCoopRep1
        if tFac2 == "maf1":
            prop = ecamPropensitiesSingleCoopRep2
        if tFac3 == "maf1":
            prop = ecamPropensitiesSingleCoopRep3
        else:
            prop = ecamPropensitiesSingleCoop
    else:
        if tFac1 == "maf1":
            prop = ecamPropensitiesDoubleCoopRep1
        if tFac2 == "maf1":
            prop = ecamPropensitiesDoubleCoopRep2
        if tFac3 == "maf1":
            prop = ecamPropensitiesDoubleCoopRep3
        else:
            prop = ecamPropensitiesDoubleCoop
        
    for i in range(start,end):
        params = testParams[i]
        a = params[0]
        b = params[1]
        c = params[2]
        d = params[3]
        e = params[4]
        f = params[5]
        print(params)
        k_rates = ecamRates(a,b,c,d,e,f)
        miscore = simMIall(k_rates, ts, x0, m, origin, prop, ecamNmatrix, sampleTimes, method)
        testScores = np.load(fileName+".npy")
        testScores[i] = miscore
        print(testScores[i])
        np.save(fileName,testScores)
        
    print(testScores)
    np.save(fileName,testScores)
    
def particleSwarmStartECAall(tFac1, tFac2, tFac3, progress, pieces=1):
    tf1,origin,sampleTimes = scaleTSall(tFac1)    
    tf2,origin,sampleTimes = scaleTSall(tFac2)
    tf3,origin,sampleTimes = scaleTSall(tFac3)
    
    n1,_ = tf1[0].shape
    n2,_ = tf2[0].shape
    n3,_ = tf3[0].shape
    
    n = np.min([n1,n2,n3])
    ts1 = [tf1[0][:n, :], tf2[0][:n, :], tf3[0][:n, :]]
    ts2 = [tf1[1][:n, :], tf2[1][:n, :], tf3[1][:n, :]]
    ts3 = [tf1[2][:n, :], tf2[2][:n, :], tf3[2][:n, :]]
    ts = [ts1, ts2, ts3]
    
    x0 = [1,0,0,0,0]

    
    for i in range(len(ts)):
        origin[i] = origin[i] - 1
    
    m = np.min([n,100])
    print(m)
    
    for i in range(len(sampleTimes)):
        sampleTimes[i] = sampleTimes[i] - sampleTimes[i][0,0]

    
    lattice = np.load("latticeExCompetingActivator.npy")
    
    startParticleSwarm('ExCompetingActivator', lattice, [[1,800],[1,800],[1,800],[1,800],[1,800],[1,800]],
                       100000, simMIall, 10, ts, x0, m, origin, ecamPropensities, ecamNmatrix,
                       sampleTimes, pieces, progress)

def particleSwarmContinueECAall(tFac1, tFac2, tFac3, maxits, pieces=1):
    tf1,origin,sampleTimes = scaleTSall(tFac1)    
    tf2,origin,sampleTimes = scaleTSall(tFac2)
    tf3,origin,sampleTimes = scaleTSall(tFac3)
    
    n1,_ = tf1[0].shape
    n2,_ = tf2[0].shape
    n3,_ = tf3[0].shape
    
    n = np.min([n1,n2,n3])
    ts1 = [tf1[0][:n, :], tf2[0][:n, :], tf3[0][:n, :]]
    ts2 = [tf1[1][:n, :], tf2[1][:n, :], tf3[1][:n, :]]
    ts3 = [tf1[2][:n, :], tf2[2][:n, :], tf3[2][:n, :]]
    ts = [ts1, ts2, ts3]
    
    x0 = [1,0,0,0,0]
        
    for i in range(len(ts)):
        origin[i] = origin[i] - 1
    

    m = np.min([n,100])
    print(m)
    
    progress = np.load("progressExCompetingActivator.npy")
    lattice = np.load("latticeExCompetingActivator.npy")
    bestPos = np.load("bestPosExCompetingActivator.npy")
    bestVals = np.load("bestValsExCompetingActivator.npy")
    swarmPos = np.load("swarmPosExCompetingActivator.npy")
    swarmVal = 1*np.load("swarmValsExCompetingActivator.npy")
    currentPos = np.load("currentPosExCompetingActivator.npy")
    currentVals = np.load("currentValsExCompetingActivator.npy")
    velocities = np.load("velocitiesExCompetingActivator.npy")
    parDict = json.load(open("parDictExCompetingActivator.json"))
    
    modelName = 'ExCompetingActivator'
    
    continueParticleSwarm(lattice, parDict, currentPos, currentVals, bestPos,
                              bestVals, swarmPos, swarmVal, velocities, progress,
                              modelName, simMIall, maxits, ts, x0, m, origin, 
                              ecamPropensities, ecamNmatrix, sampleTimes, pieces)
    
def fn():
    particleSwarmContinueECAall("dot6","dot6","dot6",maxits=8)