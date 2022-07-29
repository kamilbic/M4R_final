import json
import numpy as np

def function(a,b):
    return a*b

def parHash(pars):
    hashVal = ''
    for elements in pars:
        hashVal = hashVal + str(elements) + 'x'
    
    return hashVal

#nearest neightbour function
def nearestNeighbour(point, grid):
    dist = [np.linalg.norm(point - gridpoint) for gridpoint in grid]
    
    nnindex = np.argmin(dist)
    
    nn = grid[nnindex]
    
    return nn

def continueParticleSwarm(lattice, parDict, currentPos, currentVals, bestPos,
                          bestVals, swarmPos, swarmVal, velocities, progress, modelName, function, maxits, 
                          ts, x0, m, origin, 
                          Propensities, Nmatrix, sampleTimes, pieces):

    pos = currentPos
    vals = currentVals
    nparticles = currentPos.shape[0]
    iterations = 1*np.load("iterations{}.npy".format(modelName))
    w = 0.2
    phip = 1.5
    phig = 1.5
    
    while iterations <= maxits:
        print('iteration:', iterations)
        if progress.mean() == 1:
            progress = np.zeros(progress.shape)
        else:
            pass
        for i in range(nparticles):
            if progress[i] == 1:
                pass
            else:
                print('particle:',i+1)
                for j in range(lattice.shape[1]):
                    rp = np.random.rand(1)
                    rg = np.random.rand(1)
                    velocities[i][j] = w*velocities[i][j] + phip*rp*(bestPos[i][j] - pos[i][j]) + phig*rg*(swarmPos[j] - pos[i][j])
                
                    pos[i][j] = int(np.round(pos[i][j] + velocities[i][j]))
                    
                pos[i] = nearestNeighbour(pos[i], lattice)
                phash = parHash(pos[i])
                
                print('params:',phash)
                
                if phash in parDict:
                    val = parDict[phash]
                else:
                    val = function(pos[i],ts, x0, m, origin, 
                                   Propensities, Nmatrix, sampleTimes, pieces)
                    parDict[phash] = val
                
                vals[i] = val
                print('mi:', val)
                
                if vals[i] > bestVals[i]:
                    bestVals[i] = vals[i]
                    bestPos[i] = pos[i]
                if vals[i] > swarmVal:
                    swarmVal = vals[i]
                    swarmPos = pos[i]
                    
                progress[i] = 1
                
                np.save("iterations{}".format(modelName),iterations)
                np.save("progress{}".format(modelName),progress)
                np.save("lattice{}".format(modelName),lattice)
                np.save("bestVals{}".format(modelName),bestVals)
                np.save("bestPos{}".format(modelName),bestPos)
                np.save("swarmVals{}".format(modelName),swarmVal)
                np.save("swarmPos{}".format(modelName),swarmPos)
                np.save("currentVals{}".format(modelName),currentVals)
                np.save("currentPos{}".format(modelName),currentPos)
                json.dump(parDict,open("parDict{}.json".format(modelName),'w'))
                

        iterations += 1
        
def startParticleSwarm(modelName, lattice, param_bounds, npoints,
                   function,nparticles,ts, x0, m, origin, 
                   Propensities, Nmatrix, sampleTimes, pieces, progress):
    parDict = {}
    
    bounds = param_bounds
    
    pos = np.zeros((nparticles,len(param_bounds)))
    vals = np.zeros(nparticles)
    velocities = np.zeros(pos.shape)
    
    bestPos = np.zeros(pos.shape)
    bestVals = np.zeros(nparticles)
    
    swarmPos = np.zeros(len(param_bounds))
    swarmVal = -100000
    
    iterations = 1

    for i in range(nparticles):
        if progress[i] == 1:
            pass
        else:
            pars = [float(np.random.randint(bound[0],bound[1])) for bound in param_bounds]
            print("pars:",pars)
            pars = nearestNeighbour(pars, lattice)
            phash = parHash(pars)
            print('params:',phash)
            
            if phash in parDict:
                val = parDict[phash]
            else:
                val = function(pars,ts, x0, m, origin, 
                                   Propensities, Nmatrix, sampleTimes, pieces)
                parDict[phash] = val
            
            print('mi:', val)
            pos[i] = pars
            
            vals[i] = val
            velocities[i] = [np.random.randint(bound[0] - bound[1], 
                            bound[1] - bound[0]) for bound in bounds]
            
            bestPos[i] = pars
            bestVals[i] = val
            
            if val > swarmVal:
                swarmVal = val
                swarmPos = pars
    
            progress[i] = 1
            
            np.save("iterations{}".format(modelName),iterations)
            np.save("progress{}".format(modelName),progress)
            np.save("lattice{}".format(modelName),lattice)
            np.save("bestVals{}".format(modelName),bestVals)
            np.save("bestPos{}".format(modelName),bestPos)
            np.save("swarmVals{}".format(modelName),swarmVal)
            np.save("swarmPos{}".format(modelName),swarmPos)
            np.save("currentVals{}".format(modelName),vals)
            np.save("currentPos{}".format(modelName),pos)
            np.save("velocities{}".format(modelName),velocities)
            json.dump(parDict,open("parDict{}.json".format(modelName),'w'))