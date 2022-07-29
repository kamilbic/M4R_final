from PreCalcExtrande import Extrande
#import PreCalcExtrande
#import ExtrandeGen
from DataProcess import scaleTS
from DataProcess import scaleTSall
from MIdecodingSimp import estimateMIS
from MIexactEstimate import mi
from SimpleModelFunctions import *
from ParticleSwarmContinual import startParticleSwarm
from ParticleSwarmContinual import continueParticleSwarm
from PiecewiseLinear import piecewiselinear
from PiecewiseLinear import collapsepl
import numpy as np
import time
import json

def simMI(k_rates, ts, x0, m, origin, smPropensities, smNmatrix, sampleTimes, pieces):
    origin = origin - 1
    sm0 = sampleTimes[0,0]
    sampleTimes = sampleTimes - sm0
    sampleTimes = piecewiselinear(sampleTimes, pieces)
    
    resultsCol = np.zeros((m,len(ts[0,:])))
    RNA_resultsCol = np.zeros((m,len(ts[0,:])))
    
    ts = piecewiselinear(ts, pieces)

    results = np.zeros(sampleTimes.shape)
    RNA_results = np.zeros(sampleTimes.shape)
    
    for i in range(m):
        series = ts[i,:]
        samTimes = sampleTimes[i,:]
        
        max_t = samTimes[-1]
        times,res,t_ints,state, RNA_state = Extrande(k_rates,x0,max_t,series,len(series),smPropensities, smNmatrix, samTimes)
        results[i,:] = state[1]
        RNA_results[i,:] = RNA_state
        
        state = np.reshape(state[1], (1,-1))
        resultsCol[i,:] = collapsepl(state, pieces)
        rnstate = np.reshape(RNA_state, (1,-1))
        RNA_resultsCol[i,:] = collapsepl(rnstate, pieces)
        
     
    rich_results = results[:,origin*pieces-20*pieces:origin*pieces]
    stress_results = results[:,origin*pieces:origin*pieces+20*pieces]

    #RNA_rich_results = RNA_resultsCol[:,origin-20:origin]
    #RNA_stress_results = RNA_resultsCol[:,origin:origin+20]
    
    
    miscore = estimateMIS([[stress_results,rich_results]])[0]
    #RNA_mi = estimateMIS([RNA_stress_results,RNA_rich_results])
    
    return miscore

def MIcomparisonSingle(transcriptionFactor1, method="svm", start=0, pieces=1):
    tf,origin,sampleTimes = scaleTS(transcriptionFactor1)    
    
    n,_ = tf.shape
    
    sampleTimes = sampleTimes - sampleTimes[0,0]
    
    ts = tf
    
    x0 = [1,0,0]
    m = n
    testParams = np.load("TestLatticeSimple.npy")
    
    filename = transcriptionFactor1 + "score" + "Simple"
    
    if start == 0:
        testScores = np.zeros((50,1))
        np.save(filename,testScores)
    else:
        testScores = np.load(filename+".npy")
        start = len(testScores[testScores>0])
    
    for i in range(start,50):
        params = testParams[i]
        a = params[0]
        b = params[1]
        print(params)
        k_rates = smRates(a,b)
        miscore = simMI(k_rates, ts, x0, m, origin, 
                  smPropensities, smNmatrix, sampleTimes, 
                  pieces)
        testScores = np.load(filename+".npy")
        testScores[i] = miscore
        print(testScores[i])
        np.save(filename,testScores)
        
    print(testScores)
    np.save(filename,testScores)



def simMIDouble(k_rates, ts, x0, m, origin, Propensities, Nmatrix, sampleTimes, pieces, method):
    tstart = time.time()
    origin = origin - 1
    sm0 = sampleTimes[0,0]
    sampleTimes = sampleTimes - sm0
    sampleTimes = piecewiselinear(sampleTimes, pieces)
    
    tf1 = ts[0]
    tf2 = ts[1]

    results = np.zeros((m,len(ts[0][0,:])))
    
    statesres = 0       
    for i in range(m):
        series = [tf1[i,:], tf2[i,:]]
        samTimes = sampleTimes[i,:]
        
        max_t = samTimes[-1]
        times,res,t_ints,state,mrna = Extrande(k_rates,x0,max_t,series,len(series),Propensities, Nmatrix, samTimes)
        results[i,:] = state[1]
        statesres += state/m        
     
    rich_results = results[:,origin*pieces-20*pieces:origin*pieces]
    stress_results = results[:,origin*pieces:origin*pieces+20*pieces]

    #RNA_rich_results = RNA_resultsCol[:,origin-20:origin]
    #RNA_stress_results = RNA_resultsCol[:,origin:origin+20]
    
    if method == 'svm':
        mi = estimateMIS([[stress_results,rich_results]])[0]
        #RNA_mi = estimateMIS([RNA_stress_results,RNA_rich_results])
        
    else:
        mi = estimateMIS([[stress_results,rich_results]])[0]
        
    return mi


def MIcomparisonDouble(transcriptionFactor1, method="svm", start=0, pieces=1):
    tf,origin,sampleTimes = scaleTS(transcriptionFactor1)    
    
    n,_ = tf.shape
    
    sampleTimes = sampleTimes - sampleTimes[0,0]
    
    ts = [tf]
    
    x0 = [1,0,0]
    m = n
    testParams = np.load("TestLatticeSimple.npy")
    
    filename = transcriptionFactor1 + "score" + "Simple"
    
    if start == 0:
        testScores = np.zeros((50,1))
        np.save(filename,testScores)
    else:
        testScores = np.load(fileName+".npy")
        start = len(testScores[testScores>0])
    
    for i in range(start,50):
        params = testParams[i]
        a = params[0]
        b = params[1]
        print(params)
        k_rates = smRates(a,b)
        miscore = simMIDouble(k_rates, ts, x0, m, origin, 
                  smPropensitiesDouble, smNmatrix, sampleTimes, 
                  pieces, method)
        testScores = np.load(fileName+".npy")
        testScores[i] = miscore
        print(testScores[i])
        np.save(fileName,testScores)
        
    print(testScores)
    np.save(fileName,testScores)



def simpleMI(params, ts, x0, m, origin, Propensities, Nmatrix, 
             sampleTimes, pieces):
    
    tstart = time.time()
    a = params[0]
    b = params[1]
    krates = smRates(a,b)
    MI = simMI(krates, ts, x0, m, origin, Propensities, Nmatrix, sampleTimes, pieces)
    tend = time.time()
    print("Time Taken: {}".format((tend-tstart)/60))
    return MI


def extrandeTimes(params, ts, x0, m, origin, Propensities, Nmatrix, 
             sampleTimes, pieces=1):
    
    a = params[0]
    b = params[1]
    krates = smRates(a,b)
    
    tstartEx = time.time()
    simExtrande(krates, ts, x0, m, origin, Propensities, Nmatrix, sampleTimes, pieces)
    tendEx = time.time()
    
    tstartPC = time.time()
    simExtrandePC(krates, ts, x0, m, origin, Propensities, Nmatrix, sampleTimes, pieces)
    tendPC = time.time()
    
    timeEx = tendEx - tstartEx
    timeExPC = tendPC - tstartPC
    
    return [timeEx,timeExPC]

def timeComparison(tf,pieces=1):
    tf,origin,sampleTimes = scaleTS(tf)    
    sampleTimes = sampleTimes - sampleTimes[0,0]
    
    ts = tf[:]

    x0 = [1,0,0]
    m,_ = ts.shape
    
    paramlist = [[10**(i/2),10**(i/2)] for i in range(8)]
    times = np.zeros((8,2))
    
    for i in range(8):
        params = paramlist[i]
        times[i] = extrandeTimes(params, ts, x0, m, origin, smPropensities, smNmatrix, 
                     sampleTimes)
        print(times[i])
    
    print(times)
    np.save("ExtrandeTimeResults",times)
    
def particleSwarmStart(tf, pieces, progress):
    tf,origin,sampleTimes = scaleTS(tf)    
    sampleTimes = sampleTimes - sampleTimes[0,0]
    
    ts = tf[:]

    x0 = [1,0,0]
    m,_ = ts.shape
    
    lattice = np.load("latticeSimpleModel.npy")
    nparticles = 15
    startParticleSwarm('SimpleModel', lattice, [[1,100],[1,100]],50,simpleMI,nparticles,ts, x0,
                                      m, origin, smPropensities, smNmatrix,
                                      sampleTimes, pieces, progress)
    
def particleSwarmContinue(tf, pieces, maxits):
    tf,origin,sampleTimes = scaleTS(tf)    
    sampleTimes = sampleTimes - sampleTimes[0,0]
    
    ts = tf[:]

    x0 = [1,0,0]
    m,_ = ts.shape
    
    progress = np.load("progressSimpleModel.npy")
    lattice = np.load("latticeSimpleModel.npy")
    bestPos = np.load("bestPosSimpleModel.npy")
    bestVals = np.load("bestValsSimpleModel.npy")
    swarmPos = np.load("swarmPosSimpleModel.npy")
    swarmVal = 1*np.load("swarmValsSimpleModel.npy")
    currentPos = np.load("currentPosSimpleModel.npy")
    currentVals = np.load("currentValsSimpleModel.npy")
    velocities = np.load("velocitiesSimpleModel.npy")
    parDict = json.load(open("parDictSimpleModel.json"))
    
    modelName = 'SimpleModel'
    
    continueParticleSwarm(lattice, parDict, currentPos, currentVals, bestPos,
                              bestVals, swarmPos, swarmVal, velocities, modelName, simpleMI, 
                              maxits, ts, x0, m, origin, 
                              smPropensities, smNmatrix, sampleTimes, pieces)
    
    
def optimalB(transcriptionFactor,kon, method="svm", pieces=1):
    blow = np.linspace(kon/3,kon,6)
    bhigh = np.linspace(kon,3*kon,6)
    
    bs = np.r_[blow[:-1],bhigh]
    scores = np.zeros(bs.shape)
    
    tf,origin,sampleTimes = scaleTS(transcriptionFactor)    
    
    n,_ = tf.shape
    
    sampleTimes = sampleTimes - sampleTimes[0,0]
    
    ts = tf
    
    x0 = [1,0,0]
    m = n
    
    for i in range(len(bs)):
        b = bs[i]
        krates = smRates(kon,b)
        scores[i] = simMI(krates, ts, x0, m, origin, 
                  smPropensities, smNmatrix, sampleTimes, 
                  pieces)
        print(scores[i], b)
        
    maxb = bs[np.argmax(scores)]
    maxmi = np.max(scores)
    return maxb, maxmi
        
def optimalBs(tF,l,h,k):
    kons = 10**(np.linspace(l,h,k))
    koffs = np.zeros(kons.shape)
    scores = np.zeros(kons.shape)
    fileName = tF + "Simple" + str(l) + "and" + str(h)
    for i in range(len(kons)):
        kon = kons[i]
        print(kon)
        koff, maxmi = optimalB(tF,kon)
        koffs[i] = koff
        scores[i] = maxmi
        
        scoreArray = np.array([koffs,scores])
    
        np.save(fileName,scoreArray)
        
    return koffs, scores


def MIcomparisonAllSingle(transcriptionFactor1, method="svm", start=0, pieces=1):
    tf,origin,sampleTimes = scaleTSall(transcriptionFactor1)    
    
    n,_ = tf[0].shape
    
    ts1 = [tf[0][:n, :], np.ones(tf[0].shape)]
    ts2 = [tf[1][:n, :], np.ones(tf[0].shape)]
    ts3 = [tf[2][:n, :], np.ones(tf[0].shape)]
    ts = [ts1, ts2, ts3]
    x0 = [1,0,0]
    m = n
    
    testParams = np.load("TestLatticeSimple.npy")
    
    fileName = transcriptionFactor1 + "all" + "Simple" + method
    
    if start == 0:
        testScores = np.zeros((50,1))
        np.save(fileName,testScores)
        end = 10
    else:
        testScores = np.load(fileName+".npy")
        if start < 10 and start > 0:
            start = sum(testScores[:10]>0)[0]
            end = 10
        elif start < 20 and start >= 10:
            start = sum(testScores[10:20]>0)[0] + 10
            end = 20
        elif start < 30 and start >= 20:
            start = sum(testScores[20:30]>0)[0] + 20
            end = 30
        elif start < 40 and start >= 30:
            start = sum(testScores[30:40]>0)[0] + 30
            end = 40
        else:
            start = sum(testScores[40:]>0)[0] + 40
            end = 50
    
    
    for i in range(start,end):
        print(i)
        params = testParams[i]
        a = params[0]
        b = params[1]
        print(params)
        k_rates = smRates(a,b)
        miscore= simMIallDouble(k_rates, ts, x0, m, origin, smPropensitiesDouble, smNmatrix, sampleTimes, method)
        print(miscore)
        testScores = np.load(fileName+".npy")    
        testScores[i] = miscore
        np.save(fileName,testScores)
        
    print(testScores)
    np.save(fileName,testScores)

def simMIallDouble(k_rates, ts, x0, m, origin, Propensities, Nmatrix, sTimes, method):
    tstart = time.time()
    
    results = [np.zeros((m,len(ts[i][0][0,:]))) for i in range(len(ts))]

    stress_results = [0]*len(ts)
    rich_results = [0]*len(ts)
     
    k_rates = smRates(k_rates[0],k_rates[1])
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
def MIcomparisonAllDouble(transcriptionFactor1, transcriptionFactor2, method="svm", start=0, pieces=1):
    tf1,origin,sampleTimes = scaleTSall(transcriptionFactor1)    
    tf2,origin,sampleTimes = scaleTSall(transcriptionFactor2)
    
    n1,_ = tf1[0].shape
    n2,_ = tf2[0].shape
    
    n = np.min([n1,n2])
    ts1 = [tf1[0][:n, :], tf2[0][:n, :]]
    ts2 = [tf1[1][:n, :], tf2[1][:n, :]]
    ts3 = [tf1[2][:n, :], tf2[2][:n, :]]
    ts = [ts1, ts2, ts3]
    x0 = [1,0,0]
    m = n
    
    testParams = np.load("TestLatticeSimple.npy")
    
    fileName = transcriptionFactor1 + transcriptionFactor2 + "all" + "Simple" + "coop"
    
    if start == 0:
        testScores = np.zeros((50,1))
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
        elif start < 40 and start >= 35:
            start = sum(testScores[35:40]>0)[0] + 35
            end = 35
        elif start < 45 and start >= 40:
            start = sum(testScores[40:45]>0)[0] + 40
            end = 45
        else:
            start = sum(testScores[45:]>0)[0] + 45
            end = 50
    
    
    for i in range(start,end):
        print(i)
        params = testParams[i]
        a = params[0]
        b = params[1]
        print(params)
        k_rates = smRates(a,b)
        miscore= simMIallDouble(k_rates, ts, x0, m, origin, smPropensitiesDoubleCoop, smNmatrix, sampleTimes, method)
        print(miscore)
        testScores = np.load(fileName+".npy")    
        testScores[i] = miscore
        np.save(fileName,testScores)
        
    print(testScores)
    np.save(fileName,testScores)
    
def optimalBall(transcriptionFactor,kon, method="svm", pieces=1):
    blow = np.linspace(kon/3,kon,6)
    bhigh = np.linspace(kon,3*kon,6)
    
    bs = np.r_[blow[:-1],bhigh]
    scores = np.zeros(bs.shape)
    
    tf,origin,sampleTimes = scaleTSall(transcriptionFactor)    
    
    n,_ = tf[0].shape
    
    ts1 = [tf[0][:n, :], np.ones(tf[0].shape)]
    ts2 = [tf[1][:n, :], np.ones(tf[0].shape)]
    ts3 = [tf[2][:n, :], np.ones(tf[0].shape)]
    ts = [ts1, ts2, ts3]
    x0 = [1,0,0]
    m = n
    
    for i in range(len(sampleTimes)):
        sampleTimes[i] = sampleTimes[i] - sampleTimes[i][0,0]
    
    ts = [ts1,ts2,ts3]
    
    x0 = [1,0,0]
    
    for i in range(len(bs)):
        b = bs[i]
        krates = smRates(kon,b)
        scores[i] = simMIallDouble(krates, ts, x0, m, origin, 
                  smPropensitiesDouble, smNmatrix, sampleTimes, 
                  pieces)
        print(scores[i], b)
        
    maxb = bs[np.argmax(scores)]
    maxmi = np.max(scores)
    return maxb, maxmi
        
def optimalBsall(tF,l,h,k):
    kons = 10**(np.linspace(l,h,k))
    koffs = np.zeros(kons.shape)
    scores = np.zeros(kons.shape)
    fileName = tF +"SimpleAll"+ str(l)+"and"+str(h)
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


def particleSwarmStartSimpleall(transcriptionFactor1, progress, pieces=1):
    tf1,origin,sampleTimes = scaleTSall(transcriptionFactor1)    
    
    n,_ = tf1[0].shape
    tf2 = [np.ones(tf1[i].shape) for i in range(len(tf1))]
    
    ts1 = [tf1[0][:n, :], tf2[0][:n, :]]
    ts2 = [tf1[1][:n, :], tf2[1][:n, :]]
    ts3 = [tf1[2][:n, :], tf2[2][:n, :]]
    ts = [ts1, ts2, ts3]
    x0 = [1,0,0]
    m = np.min([n,100])
    print(m)
        
    for i in range(len(ts)):
        origin[i] = origin[i] - 1
    
    for i in range(len(sampleTimes)):
        sampleTimes[i] = sampleTimes[i] - sampleTimes[i][0,0]

    
    lattice = np.load("latticeSimpleModel.npy")
    
    startParticleSwarm('SimpleModel', lattice, [[1,800],[1,800]],
                       4000, simMIallDouble, 10, ts, x0, m, origin, smPropensitiesDouble, smNmatrix,
                       sampleTimes, pieces, progress)

def particleSwarmContinueSimpleall(transcriptionFactor1,maxits, pieces=1):
    tf1,origin,sampleTimes = scaleTSall(transcriptionFactor1)    
    
    n,_ = tf1[0].shape
    tf2 = [np.ones(tf1[i].shape) for i in range(len(tf1))]
    
    ts1 = [tf1[0][:n, :], tf2[0][:n, :]]
    ts2 = [tf1[1][:n, :], tf2[1][:n, :]]
    ts3 = [tf1[2][:n, :], tf2[2][:n, :]]
    ts = [ts1, ts2, ts3]
    x0 = [1,0,0]
    m = np.min([n,100])
    print(m)
        
    for i in range(len(ts)):
        origin[i] = origin[i] - 1
    
    progress = np.load("progressSimpleModel.npy")
    lattice = np.load("latticeSimpleModel.npy")
    bestPos = np.load("bestPosSimpleModel.npy")
    bestVals = np.load("bestValsSimpleModel.npy")
    swarmPos = np.load("swarmPosSimpleModel.npy")
    swarmVal = 1*np.load("swarmValsSimpleModel.npy")
    currentPos = np.load("currentPosSimpleModel.npy")
    currentVals = np.load("currentValsSimpleModel.npy")
    velocities = np.load("velocitiesSimpleModel.npy")
    parDict = json.load(open("parDictSimpleModel.json"))
    
    modelName = 'SimpleModel'
    
    continueParticleSwarm(lattice, parDict, currentPos, currentVals, bestPos,
                              bestVals, swarmPos, swarmVal, velocities, progress,
                              modelName, simMIallDouble, maxits, ts, x0, m, origin, 
                              smPropensitiesDouble, smNmatrix, sampleTimes, pieces)
    
def fn():
    prog = np.load("progressSimpleModel.npy")
    particleSwarmStartSimpleall("dot6",prog)
    particleSwarmContinueSimpleall("dot6",maxits=8)
    