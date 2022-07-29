from PreCalcExtrande import Extrande
from DataProcess import scaleTS
from DataProcess import scaleTSall
from MIdecodingSimp import estimateMIS
from MIexactEstimate import mi
from DerepressionModelFunctions import *
from ParticleSwarmContinual import startParticleSwarm
from ParticleSwarmContinual import continueParticleSwarm
from PiecewiseLinear import piecewiselinear
from PiecewiseLinear import collapsepl
import numpy as np
import time
import json
from matplotlib import pyplot as plt


def simMI(k_rates, ts, x0, m, origin, Propensities, Nmatrix, sampleTimes, pieces, method='svm'):
    tstart = time.time()
    origin = origin - 1
    sm0 = sampleTimes[0,0]
    sampleTimes = sampleTimes - sm0
    sampleTimes = piecewiselinear(sampleTimes, pieces)
    
    resultsColA = np.zeros((m,len(ts[0][0,:])))
    resultsColB = np.zeros((m,len(ts[0][0,:])))
    
    tf1 = ts[0]
    tf2 = ts[1]
    
    tf1 = piecewiselinear(tf1, pieces)
    tf2 = piecewiselinear(tf2, pieces)

    
    resultsA = np.zeros((m,len(ts[0][0,:])))
    resultsB = np.zeros((m,len(ts[0][0,:])))
    
    results = np.zeros((m,2,len(ts[0][0,:])))
    
    statesres = 0       
    k_rates = dmRates(k_rates[0],k_rates[1],k_rates[2],k_rates[3])
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
    
    testParams = np.load("TestLattice.npy")
    
    fileName = transcriptionFactor1 + transcriptionFactor2 + "score" + "DerepressionRepressor" + method
    
    if start == 0:
        testScores = np.zeros((40,1))
    else:
        testScores = np.load(fileName+".npy")
        start = len(testScores[testScores>0])
    
    for i in range(start,40):
        params = testParams[i]
        a = params[0]
        b = params[1]
        c = params[2]
        d = params[3]
        print(params)
        k_rates = dmRates(a,b,c,d)
        testScores[i] = simMI(k_rates, ts, x0, m, origin, dmPropensities, dmNmatrix, sampleTimes, pieces, method)
        print(testScores[i])
        np.save(fileName,testScores)
        
    print(testScores)
    np.save(fileName,testScores)
    
def simExtrande(k_rates, ts, x0, m, origin, Propensities, Nmatrix, 
             sampleTimes, pieces=1):

    origin = origin - 1
    sm0 = sampleTimes[0,0]
    sampleTimes = sampleTimes - sm0
    sampleTimes = piecewiselinear(sampleTimes, pieces)
    
    tf1 = ts[0]
    tf2 = ts[1]
    
    tf1 = piecewiselinear(tf1, pieces)
    tf2 = piecewiselinear(tf2, pieces)

    k_rates = dmRates(k_rates[0],k_rates[1],k_rates[2],k_rates[3])
    for i in range(m):

        series = [tf1[i,:], tf2[i,:]]
        samTimes = sampleTimes[i,:]
        
        max_t = samTimes[-1]
        times,res,t_ints,state,mrna = ExtrandeGen.Extrande(k_rates,x0,max_t,series,len(series),Propensities, Nmatrix, samTimes)
    
def simExtrandePC(k_rates, ts, x0, m, origin, Propensities, Nmatrix, 
             sampleTimes, pieces=1):

    origin = origin - 1
    sm0 = sampleTimes[0,0]
    sampleTimes = sampleTimes - sm0
    sampleTimes = piecewiselinear(sampleTimes, pieces)
    
    tf1 = ts[0]
    tf2 = ts[1]
    
    tf1 = piecewiselinear(tf1, pieces)
    tf2 = piecewiselinear(tf2, pieces)

    k_rates = dmRates(k_rates[0],k_rates[1],k_rates[2],k_rates[3])
    for i in range(m):

        series = [tf1[i,:], tf2[i,:]]
        samTimes = sampleTimes[i,:]
        
        max_t = samTimes[-1]
        times,res,t_ints,state,mrna = PreCalcExtrande.Extrande(k_rates,x0,max_t,series,len(series),Propensities, Nmatrix, samTimes)
    


    
def extrandeTimes(params, ts, x0, m, origin, Propensities, Nmatrix, 
             sampleTimes, pieces=1):
    
    tstartEx = time.time()
    simExtrande(params, ts, x0, m, origin, Propensities, Nmatrix, sampleTimes, pieces)
    tendEx = time.time()
    
    tstartPC = time.time()
    simExtrandePC(params, ts, x0, m, origin, Propensities, Nmatrix, sampleTimes, pieces)
    tendPC = time.time()
    
    timeEx = tendEx - tstartEx
    timeExPC = tendPC - tstartPC
    
    return [timeEx,timeExPC]

def timeComparison(transfact1,transfact2,pieces=1):
    tf1,origin,sampleTimes = scaleTS(transfact1)
    tf2,origin,sampleTimes = scaleTS(transfact2)    
    sampleTimes = sampleTimes - sampleTimes[0,0]
    
    n1,m = tf1.shape
    n2,_ = tf2.shape
    
    sampleTimes = sampleTimes - sampleTimes[0,0]
    
    n = np.min([n1,n2])
    ts = [tf1[:n, :], tf2[:n, :]]

    x0 = [1,0,0,0]
    
    paramlist = [[10**(i/2),10**(i/2),10**(i/2),10**(i/2)] for i in range(7)]
    times = np.zeros((7,2))
    
    for i in range(7):
        params = paramlist[i]
        times[i] = extrandeTimes(params, ts, x0, m, origin, dmPropensities, dmNmatrix, 
                     sampleTimes)
        print(times[i])
    
    print(times)
    np.save("ExtrandeTimeResults3State",times)
    
def particleSwarmStartD(transcriptionFactor1, transcriptionFactor2, progress, pieces=1):
    tf1,origin,sampleTimes = scaleTS(transcriptionFactor1)    
    tf2,origin,sampleTimes = scaleTS(transcriptionFactor2)
    
    n1,_ = tf1.shape
    n2,_ = tf2.shape
    
    sampleTimes = sampleTimes - sampleTimes[0,0]
    
    n = np.min([n1,n2])
    ts = [tf1[:n, :], tf2[:n, :]]
    
    x0 = [1,0,0,0]
    m = n
    
    lattice = np.load("latticeDerepression.npy")
    
    startParticleSwarm('Derepression', lattice, [[1,100],[1,100],[1,100],[1,100]],10000,
                       simMI,15,ts, x0, m, origin, dmPropensities, dmNmatrix,
                       sampleTimes, pieces, progress)

def particleSwarmContinueD(transcriptionFactor1, transcriptionFactor2, pieces, maxits):
    tf1,origin,sampleTimes = scaleTS(transcriptionFactor1)    
    tf2,origin,sampleTimes = scaleTS(transcriptionFactor2)
    
    n1,_ = tf1.shape
    n2,_ = tf2.shape 
    
    sampleTimes = sampleTimes - sampleTimes[0,0]
    
    n = np.min([n1,n2])
    ts = [tf1[:n, :], tf2[:n, :]]
    
    x0 = [1,0,0,0]
    m = n
    
    progress = np.load("progressDerepression.npy")
    lattice = np.load("latticeDerepression.npy")
    bestPos = np.load("bestPosDerepression.npy")
    bestVals = np.load("bestValsDerepression.npy")
    swarmPos = np.load("swarmPosDerepression.npy")
    swarmVal = 1*np.load("swarmValsDerepression.npy")
    currentPos = np.load("currentPosDerepression.npy")
    currentVals = np.load("currentValsDerepression.npy")
    velocities = np.load("velocitiesDerepression.npy")
    parDict = json.load(open("parDictDerepression.json"))
    
    modelName = 'Derepression'
    
    continueParticleSwarm(lattice, parDict, currentPos, currentVals, bestPos,
                              bestVals, swarmPos, swarmVal, velocities, progress,
                              modelName, simMI, maxits, ts, x0, m, origin, 
                              dmPropensities, dmNmatrix, sampleTimes, pieces)
    
def simMIall(k_rates, ts, x0, m, origin, Propensities, Nmatrix, sTimes, method):
    tstart = time.time()
    
    resultsA = [np.zeros((m,len(ts[i][0][0,:]))) for i in range(len(ts))]
    resultsB = [np.zeros((m,len(ts[i][0][0,:]))) for i in range(len(ts))]
    
    stress_results = [0]*len(ts)
    rich_results = [0]*len(ts)
     
    k_rates = dmRates(k_rates[0],k_rates[1],k_rates[2],k_rates[3])
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
    x0 = [1,0,0,0]
    m = n
    
    testParams = np.load("TestLattice.npy")
    
    fileName = transcriptionFactor1 + transcriptionFactor2 + "all" + "Derepression" + method
    
    if start == 0:
        testScores = np.zeros((40,1))
    else:
        testScores = np.load(fileName+".npy")
        start = len(testScores[testScores>0])
    
    for i in range(start,40):
        params = testParams[i]
        a = params[0]
        b = params[1]
        c = params[2]
        d = params[3]
        print(params)
        k_rates = dmRates(a,b,c,d)
        testScores[i] = simMIall(k_rates, ts, x0, m, origin, dmPropensities, dmNmatrix, sampleTimes, method)
        print(testScores[i])
        np.save(fileName,testScores)
        
    print(testScores)
    np.save(fileName,testScores)


def particleSwarmStartDall(transcriptionFactor1, transcriptionFactor2, progress, pieces=1):
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
        
    for i in range(len(ts)):
        origin[i] = origin[i] - 1
    
    
    for i in range(len(sampleTimes)):
        sampleTimes[i] = sampleTimes[i] - sampleTimes[i][0,0]

    
    lattice = np.load("latticeCompetingActivator.npy")
    
    startParticleSwarm('Derepression', lattice, [[1,800],[1,800],[1,800],[1,800]],
                       10000, simMIall, 10, ts, x0, m, origin, dmPropensities, dmNmatrix,
                       sampleTimes, pieces, progress)

def particleSwarmContinueDall(transcriptionFactor1, transcriptionFactor2,maxits, pieces=1):
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
        
    for i in range(len(ts)):
        origin[i] = origin[i] - 1
    
    
    progress = np.load("progressDerepression.npy")
    lattice = np.load("latticeCompetingActivator.npy")
    bestPos = np.load("bestPosDerepression.npy")
    bestVals = np.load("bestValsDerepression.npy")
    swarmPos = np.load("swarmPosDerepression.npy")
    swarmVal = 1*np.load("swarmValsDerepression.npy")
    currentPos = np.load("currentPosDerepression.npy")
    currentVals = np.load("currentValsDerepression.npy")
    velocities = np.load("velocitiesDerepression.npy")
    parDict = json.load(open("parDictDerepression.json"))
    
    modelName = 'Derepression'
    
    continueParticleSwarm(lattice, parDict, currentPos, currentVals, bestPos,
                              bestVals, swarmPos, swarmVal, velocities, progress,
                              modelName, simMIall, maxits, ts, x0, m, origin, 
                              dmPropensities, dmNmatrix, sampleTimes, pieces)
    
def fn():
    prog = np.load("progressDerepression.npy")
    particleSwarmStartDall("dot6","dot6",prog)
    particleSwarmContinueDall("dot6","dot6",maxits=8)