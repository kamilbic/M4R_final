'''
Extrande Algorithm - General
'''

import numpy as np
import random


def propensityBound(current_t, max_t, x, x_series, k_rates, freq, propFun):
    #remaining time till next interval
    t_rem = max_t - current_t
    #length of x series
    if type(x_series) == list:
        n = len(x_series[0])
    else:
        n = len(x_series)
    
    #check if simulation will go over time
    if t_rem > max_t/(n/3):
        L = max_t/(n/3)
    else:
        L = t_rem
        
    #interval at current time
    cur_obs = int(current_t // freq)
    #interval at next time
    tl_obs = int((current_t + L) // freq)
    #number of intervals which need to be checked
    num_obs = tl_obs - cur_obs
    
    if num_obs == 0:
        num_obs = 1
    
    #propensity vectors initialised
    props = [0]*num_obs
    
    #propensities calculated
    for i in range(num_obs):
        ###################
        '''
        Check below line later
        '''
        if type(x_series) == list:
            xt = [x_s[cur_obs+i] for x_s in x_series]
        else:
            xt = x_series[cur_obs+i]
        ###################
        props[i] = sum(propFun(k_rates,x,xt))
    
    #propensity bound calculated
    B = max(props)
    
    return L, B

#function to find next interval
def next_int(x,xlist):
    i = 0
    while x > xlist[i]:
        i += 1
    
    return i

#function to fill in the rest of the current interval in case 
# no changes in state occur before a change in time interval
def fill_ints(ti,tim1,tlist,istart,iend,xlist):
    xlist[iend] = ti - tlist[iend-1]
    
    for i in range(iend-istart-1):
        if i == 0:
            pass
        else:
            xlist[iend-i] += tlist[iend-i]-tlist[iend-i-1]
        
    xlist[istart] += tlist[istart] - tim1
    
    return xlist


def Extrande(ks, x0, max_t, tseries,tsteps, propFun, NMatrix, reactTimes):
    N = NMatrix()
    
    #Time intervals and Active State Fraction matrix
    #t_intervals = np.linspace(max_t/tsteps,max_t,tsteps)
    t_intervals = reactTimes
    states = np.zeros((len(x0[:-1]),len(t_intervals)))
    t_int = 0
    cur_state = 1.0*np.array(x0)
    
    RNA_state = np.zeros((len(t_intervals)))
    RNA_interval = np.zeros((len(t_intervals)))
    
    t = 0.0 #starting time
    ts = [0.0]  #reporting array, to save information
    x = np.copy(x0) #using the given initial condition
    res = [list(x)]
    
    #number of observations in experiment
    if type(tseries) == list:
        obs = len(tseries[0])
    else:
        obs = len(tseries)
    #frequency of observations
    freq = max_t/obs

    tsum = 0
    
    while True: #just continuously looping until there is a break from within
        #calculate propensity bound
        prev_t = t
        L, B = propensityBound(t,max_t,x,tseries,ks,freq,propFun)
        #calculate tau
        tau = random.expovariate(B)
        
        if tau > L:
            if t + L > max_t:
                t = max_t
            else:
                t = t + L
            ts.append(t)
            res.append(list(x))
        else:
            #which observation is observed
            tstep = int(t // freq)
            #current scaled observation value
            if type(tseries) == list:
                xt = [x_s[tstep] for x_s in tseries]
            else:
                xt = tseries[tstep]

            #propensities
            prop = propFun(ks, x, xt)
            a0 = sum(prop) #total propensity
            if a0 == 0:
              break
            
            #generate uniform random variable
            u = np.random.uniform(0,1)
            #update time
            t = t + tau
            
            if a0 >= B*u:
                propsum = 0
                for i in range(len(prop)):
                    propsum += prop[i]
                    if propsum >= B*u:
                        change_to_apply = N[i,:] #idx applied to the state vector
                        #need to re shape because it comes out as a 2D array
                        change_to_apply = np.round(np.reshape(change_to_apply,(len(x),))) #converting to 1D array
                        #How the state is going to change

                        x += change_to_apply
                        break
            else:
                #no change to system
                pass
            
            #saving the time and results so that we can use it later
            ts.append(t)
            res.append(list(x))
            
        for i in range(len(cur_state)-1):
            if x[i] == 1 and cur_state[i] == 1:
                t_int = next_int(prev_t,t_intervals)
                if t > t_intervals[t_int]:
                    next_t_int = next_int(t,t_intervals)
                    states[i] = fill_ints(t,prev_t,t_intervals,t_int,
                                             next_t_int,states[i])
                else:
                    states[i,t_int] += t-prev_t
            elif x[i] == 0 and cur_state[i] == 1:
                t_int = next_int(prev_t,t_intervals)
                if t > t_intervals[t_int]:
                    next_t_int = next_int(t,t_intervals)
                    states[i] = fill_ints(t,prev_t,t_intervals,t_int,
                                       next_t_int,states[i])
                else:
                    states[i,t_int] += t-prev_t
            else:
                pass
            
            
        if t > t_intervals[t_int]:
            t_int = next_t_int
            RNA_state[t_int] = cur_state[-1]
            RNA_interval[t_int] = RNA_state[t_int] - RNA_state[t_int-1]
        tsum += t-prev_t
            
        cur_state = 1.0*x
        
            
        if int(t) == int(max_t):
            break
        elif t > max_t:
            break
        else:
            pass
    
    for i in range(states.shape[1]-1):
        states[:,i+1] = states[:,i+1]/(t_intervals[i+1]-t_intervals[i])

    return ts, np.array(res),t_intervals,states, RNA_state
