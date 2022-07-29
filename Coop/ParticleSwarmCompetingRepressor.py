from PreCalcExtrande import Extrande
#from ExtrandeGen import Extrande
from DataProcess import scaleTS
from DataProcess import scaleTSall
from MIdecodingSimp import estimateMIS
from MIexactEstimate import mi
from CompetingRepressorModelFunctions import *
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
    
    resultsCol = np.zeros((m,len(ts[0][0,:])))
    
    tf1 = ts[0]
    tf2 = ts[1]
    
    tf1 = piecewiselinear(tf1, pieces)
    tf2 = piecewiselinear(tf2, pieces)
    
    results = np.zeros((m,len(ts[0][0,:])))
    
    statesres = 0       
    k_rates = crmRates(k_rates[0],k_rates[1],k_rates[2],k_rates[3])
    for i in range(m):

        series = [tf1[i,:], tf2[i,:]]
        samTimes = sampleTimes[i,:]
        
        max_t = samTimes[-1]
        times,res,t_ints,state,mrna = Extrande(k_rates,x0,max_t,series,len(series),Propensities, Nmatrix, samTimes)
        results[i,:] = state[1] 
        statesres += state/m
        resultsCol[i,:] = collapsepl(np.reshape(state[1],(1,-1)), pieces)    
    
    plt.figure()
    plt.plot(statesres[0][origin-20:origin+20],label='I1')
    plt.plot(statesres[1][origin-20:origin+20],label='A')
    plt.plot(statesres[2][origin-20:origin+20],label='I2')
    plt.legend(loc='upper left')
    plt.show()
    
    stress_results = results[:,origin:origin+20]
    rich_results = results[:,origin-20:origin]
    
    if method == "svm":
        MI = estimateMIS([[stress_results, rich_results]])
        ttaken = time.time() - tstart
        print("Time Taken: ", ttaken)
    
        return MI[0]
    else:
        inp = np.r_[rich_results,stress_results]
        out = np.r_[np.zeros(rich_results.shape[0]),np.ones(stress_results.shape[0])]
        MI = mi(inp,out)
        ttaken = time.time() - tstart
        print("Time Taken: ", ttaken)
    
        return MI
    
    
def MIcomparison(transcriptionFactor1, transcriptionFactor2,method="svm",start=0,pieces=1):
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
    
    fileName = transcriptionFactor1 + transcriptionFactor2 + "score" + "CompetingRepressor" + method
    
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
        k_rates = crmRates(a,b,c,d)
        testScores[i] = simMI(k_rates, ts, x0, m, origin, crmPropensities, crmNmatrix, sampleTimes, pieces, method)
        print(testScores[i])
        np.save(fileName,testScores)
        
    print(testScores)
    np.save(fileName,testScores)
    
def particleSwarmStartCR(transcriptionFactor1, transcriptionFactor2, progress, pieces=1):
    tf1,origin,sampleTimes = scaleTS(transcriptionFactor1)    
    tf2,origin,sampleTimes = scaleTS(transcriptionFactor2)
    
    n1,_ = tf1.shape
    n2,_ = tf2.shape
    
    sampleTimes = sampleTimes - sampleTimes[0,0]
    
    n = np.min([n1,n2])
    ts = [tf1[:n, :], tf2[:n, :]]
    
    x0 = [1,0,0,0]
    m = n
    
    lattice = np.load("latticeCompetingRepressor.npy")
    
    startParticleSwarm('CompetingRepressor', lattice, [[1,1000],[1,1000],[1,1000],[1,1000]],10000,
                       simMI, 15, ts, x0, m, origin, crmPropensities, crmNmatrix,
                       sampleTimes, pieces, progress)

def particleSwarmContinueCR(transcriptionFactor1, transcriptionFactor2, pieces, maxits):
    tf1,origin,sampleTimes = scaleTS(transcriptionFactor1)    
    tf2,origin,sampleTimes = scaleTS(transcriptionFactor2)
    
    n1,_ = tf1.shape
    n2,_ = tf2.shape 
    
    sampleTimes = sampleTimes - sampleTimes[0,0]
    
    n = np.min([n1,n2])
    ts = [tf1[:n, :], tf2[:n, :]]
    
    x0 = [1,0,0,0]
    m = n
    
    progress = np.load("progressCompetingRepressor.npy")
    lattice = np.load("latticeCompetingRepressor.npy")
    bestPos = np.load("bestPosCompetingRepressor.npy")
    bestVals = np.load("bestValsCompetingRepressor.npy")
    swarmPos = np.load("swarmPosCompetingRepressor.npy")
    swarmVal = 1*np.load("swarmValsCompetingRepressor.npy")
    currentPos = np.load("currentPosCompetingRepressor.npy")
    currentVals = np.load("currentValsCompetingRepressor.npy")
    velocities = np.load("velocitiesCompetingRepressor.npy")
    parDict = json.load(open("parDictCompetingRepressor.json"))
    
    modelName = 'CompetingRepressor'
    
    continueParticleSwarm(lattice, parDict, currentPos, currentVals, bestPos,
                              bestVals, swarmPos, swarmVal, velocities, progress,
                              modelName, simMI, maxits, ts, x0, m, origin, 
                              crmPropensities, crmNmatrix, sampleTimes, pieces)
    
def simMIall(k_rates, ts, x0, m, origin, Propensities, Nmatrix, sTimes, method):
    tstart = time.time()

    results = [np.zeros((m,len(ts[i][0][0,:]))) for i in range(len(ts))]

    stress_results = [0]*len(ts)
    rich_results = [0]*len(ts)
     
    k_rates = crmRates(k_rates[0],k_rates[1],k_rates[2],k_rates[3])
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
            results[j][i,:] = state[1] 
        
        orig = origin[j]
        stress_results[j] = results[j][:,orig:orig+20]
        rich_results[j] = results[j][:,orig-20:orig]
        
    
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
    
    fileName = transcriptionFactor1 + transcriptionFactor2 + "all" + "CompetingRepressor" + method
    
    if start == 0:
        testScores = np.zeros((40,1))
        np.save(fileName,testScores)
        end = 5
    else:
        testScores = np.load(fileName+".npy")
        if start < 5:
            start = sum(testScores[:5]>0)[0]
            end = 5
        elif start < 10 and start >= 5:
            start = sum(testScores[5:10]>0)[0] + 5
            end = 10
        elif start < 15 and start >= 10:
            start = sum(testScores[10:15]>0)[0] + 10
            end = 15
        elif start < 20 and start >= 15:
            start = sum(testScores[15:20]>0)[0] + 15
            end = 20
        elif start < 25 and start >= 20:
            start = sum(testScores[20:25]>0)[0] + 20
            end = 25
        elif start < 30 and start >= 25:
            start = sum(testScores[25:30]>0)[0] + 25
            end = 30
        elif start < 35 and start >= 30:
            start = sum(testScores[30:35]>0)[0] + 30
            end = 35
        elif start >= 35:
            start = sum(testScores[35:]>0)[0] + 35
            end = 40
    
    for i in range(start,end):
        params = testParams[i]
        a = params[0]
        b = params[1]
        c = params[2]
        d = params[3]
        print(params)
        k_rates = crmRates(a,b,c,d)
        miscore = simMIall(k_rates, ts, x0, m, origin, crmPropensities, crmNmatrix, sampleTimes, method)
        testScores = np.load(fileName+".npy")
        testScores[i] = miscore
        print(testScores[i])
        np.save(fileName,testScores)
        
    print(testScores)
    np.save(fileName,testScores)
    
def particleSwarmStartCRall(transcriptionFactor1, transcriptionFactor2, progress, pieces=1):
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

    
    lattice = np.load("latticeCompetingRepressor.npy")
    
    startParticleSwarm('CompetingRepressorMixed', lattice, [[1,800],[1,800],[1,800],[1,800]],
                       10000, simMIall, 10, ts, x0, m, origin, crmPropensities, crmNmatrix,
                       sampleTimes, pieces, progress)

def particleSwarmContinueCRall(transcriptionFactor1, transcriptionFactor2,maxits, pieces=1):
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
    
    
    progress = np.load("progressCompetingRepressorMixed.npy")
    lattice = np.load("latticeCompetingRepressor.npy")
    bestPos = np.load("bestPosCompetingRepressorMixed.npy")
    bestVals = np.load("bestValsCompetingRepressorMixed.npy")
    swarmPos = np.load("swarmPosCompetingRepressorMixed.npy")
    swarmVal = 1*np.load("swarmValsCompetingRepressorMixed.npy")
    currentPos = np.load("currentPosCompetingRepressorMixed.npy")
    currentVals = np.load("currentValsCompetingRepressorMixed.npy")
    velocities = np.load("velocitiesCompetingRepressorMixed.npy")
    parDict = json.load(open("parDictCompetingRepressorMixed.json"))
    
    modelName = 'CompetingRepressorMixed'
    
    continueParticleSwarm(lattice, parDict, currentPos, currentVals, bestPos,
                              bestVals, swarmPos, swarmVal, velocities, progress,
                              modelName, simMIall, maxits, ts, x0, m, origin, 
                              crmPropensities, crmNmatrix, sampleTimes, pieces)
    
def fn():
    prog = np.load("progressCompetingRepressorMixed.npy")
    particleSwarmStartCRall("dot6","maf1",prog)
    particleSwarmContinueCRall("dot6","maf1",8)