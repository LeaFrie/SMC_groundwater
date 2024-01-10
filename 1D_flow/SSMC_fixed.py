# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:01:13 2022

@author: lfriedl2
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scipy
import statistics as statistics
from scipy import stats
from numpy.random import random
from joblib import Parallel, delayed
from generate_data import *


# systematic resampling
# https://filterpy.readthedocs.io/en/latest/_modules/filterpy/monte_carlo/resampling.html#systematic_resample
def systematic_resample(weights):
    """ Performs the systemic resampling algorithm used by particle filters.

    This algorithm separates the sample space into N divisions. A single random
    offset is used to to choose where to sample from for all divisions. This
    guarantees that every sample is exactly 1/N apart.

    Parameters
    ----------
    weights : list-like of float
        list of weights as floats

    Returns
    -------

    indexes : ndarray of ints
        array of indexes into the weights defining the resample. i.e. the
        index of the zeroth resample is indexes[0], etc.
    """
    N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (random() + np.arange(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def mcmc_steps(norm_old, jr):

    summa = np.zeros((nr_red,nr_space))
    for n in range(nr_red):
        summa[n,:] = norm_old[n]*np.sqrt(ww[n])*vv[:,n]
             
    ld_old = mean_ld + np.sum(summa, 0)
    y_old = forward(ld_old, s)     
    lh_old = np.sum(stats.norm.logpdf(y_old.T, y_true, obs_noise))
    prior_old = np.prod(stats.norm.pdf(norm_old))
    
    arr = np.zeros(mc_steps)
    for it in range(mc_steps):
        
        norm_new = np.zeros((10,1))
        for com in range(10):
            norm_new[com] = norm_old[com] + np.random.normal(0,jr[com],1) # np.sqrt(1-jr**2)*norm_old + np.random.normal(0,jr,10)
        norm_new = norm_new.flatten()

        summa = np.zeros((nr_red,nr_space))
        for n in range(nr_red):
            summa[n,:] = norm_new[n]*np.sqrt(ww[n])*vv[:,n]
                 
        ld_new = mean_ld + np.sum(summa, 0)
        y_new = forward(ld_new, s)     
        lh_new = np.sum(stats.norm.logpdf(y_new.T, y_true, obs_noise))
        prior_new = np.prod(stats.norm.pdf(norm_new))
        
        aa = min(1,(np.exp((lh_new-lh_old)*alpha1[t])*(prior_new/prior_old)))
        
        R = risk(ld_new)
        
        if np.random.uniform(0,1,1) < aa and R >= alpha2[t]:
            arr[it] = 1
            norm_old = norm_new 
            lh_old = lh_new
            prior_old = prior_new
            ld_old = ld_new
            
    return norm_old, lh_old, ld_old, R, arr


#%%
nr_space = 40
l_grid = 1

mean_ld = np.log(10**(-5))
sd_ld = 3
lamb = 0.3
nr_red = 10

knots = np.linspace(0,1,nr_space+1)
cell_m = np.linspace(0,l_grid, nr_space)+l_grid/(2*nr_space)

tl = scipy.loadmat('Eig.mat')
vv = tl['vv']
ww = tl['ww'].flatten()

s = 0.001  #m^3/s
obs_noise= 0.01

tl = scipy.loadmat('Data.mat')
y_true = tl['data']
y_pure = tl['data1']
norm_true = tl['norm_ld']
ld_true = tl['ld_true']
 
#%% SMC for posterior
nrp = 50 # nr of particles
cess_op = 0.9 # optimal CESS
mc_steps = 20 # MH steps
ess_res = 0.3 # resampling

# target thresholds
threshold1 = 0.9e-05
threshold2 = 0.95e-05

# fixed sequence of thresholds
nr_risk = 100
x = np.linspace(0,nr_risk,nr_risk)
x[0] = 1e-10
b = 5e-06
a = (threshold2-b)/np.log(nr_risk)
alpha2_fix = a*np.log(x) + b
alpha2_fix[np.argmin(np.abs(alpha2_fix - threshold1))] = threshold1
alpha2_fix[0] = 0

nr_post = 1000
nr_run = 10

probb1 = np.zeros((nr_run))
probb2 = np.zeros((nr_run))
nr = np.zeros((nr_run))

for ppp in range(nr_run):
    
    ittotal = 100000
    alpha1 = np.zeros((ittotal)) 
    alpha2 = np.zeros((ittotal)) 
    jr = np.linspace(0.006,0.1,10)
    weight = np.zeros((ittotal,nrp))
    weight_norm = np.zeros((ittotal,nrp))
    sigma = np.zeros((ittotal,nrp,10))
    sssigma = np.zeros((ittotal,nrp,10))
    conc = np.zeros((ittotal,nrp))
    pr = np.ones((ittotal,1))
    count = np.zeros((nrp,ittotal))
    acc_R = np.zeros((ittotal, nrp))
    
    # initialization
    sigma[0,:,:] = stats.norm.rvs(size=(nrp,10))
    weight[0,:] = np.linspace(1/nrp,1/nrp,nrp)
    weight_norm[0,:] = np.linspace(1/nrp,1/nrp,nrp)
    
    ld_curr = np.zeros((nrp, nr_space))
    for p in range(nrp):
        
        norm_old = sigma[0,p,:]
        
        summa = np.zeros((nr_red,nr_space))
        for n in range(nr_red):
            summa[n,:] = norm_old[n]*np.sqrt(ww[n])*vv[:,n]
                 
        ld_old = mean_ld + np.sum(summa, 0)
        risk_old = risk(ld_old)
        
        ld_curr[p,:] = ld_old
        conc[0,p] = risk_old    
    
    # start
    t = 1
    risk_ind = 0

    while alpha2[t]<threshold2:
        
        if alpha1[t-1]<1:
    
            lh_old = np.zeros((nrp)); y_old = np.zeros((nrp,7))
            for p in range(nrp):
                y_old[p,:] = forward(ld_curr[p,:], s).flatten()  
                lh_old[p] = np.sum(stats.norm.logpdf(y_old[p,:].flatten(), y_true, obs_noise))
            
            if np.mod(ppp,1)==0:   
                # binary search to find alph such that cess_alph =  cess_op      
                k = 1
                alph_incr =  (10**(-1)-10**(-10))/(2**k) 
                alpha1[t] = alpha1[t-1]+alph_incr 
                w = np.exp((alpha1[t]-alpha1[t-1])*lh_old)  
                cess = np.dot(weight_norm[t-1,:],w)**2/np.dot(weight_norm[t-1,:],w**2)
                
                while k<100: 
                    
                    if np.isnan(cess):
                        cess = 0
                    
                    k = k+1
                    
                    if cess < cess_op:
                        alph_incr =  alph_incr - (10**(-1)-10**(-10))/(2**k)
                        
                    if cess > cess_op:
                        alph_incr =  alph_incr + (10**(-1)-10**(-10))/(2**k)    
                    
                    alpha1[t] = alpha1[t-1]+alph_incr 
                    w = np.exp((alpha1[t]-alpha1[t-1])*lh_old)   
                    cess = np.dot(weight_norm[t-1,:],w)**2/np.dot(weight_norm[t-1,:],w**2)
                
                alpha1[t] = min(max(alpha1[t],0),1)
                if alpha1[t] == 1:
                    alpha1[t:ittotal] = 1
                    nr_post = t
                    mc_steps = mc_steps1
                    
        sigma[t,:,:] = sigma[t-1,:,:]
        sr = range(nrp)
        
        if  alpha1[t-1]<1:
        
            for p in range(nrp):
                  
                pdf = np.sum(stats.norm.logpdf(y_old[p,:], y_true, obs_noise))
                    
                weight[t,p] =  np.exp((alpha1[t]-alpha1[t-1])*pdf) # gamma_new/gamma_old
                   
            weight_norm[t,:] = np.multiply(weight_norm[t-1,:],weight[t, :].flatten())/np.dot(weight_norm[t-1,:],weight[t,:])
            ess = np.dot(weight_norm[t-1,:],weight[t,:])**2/np.dot(weight_norm[t-1,:]**2,weight[t,:]**2)
             
            if ess/nrp < ess_res or t == nr_post:
                    sr = systematic_resample(weight_norm[t,:])
                    weight_norm[t,:] = np.linspace(1/nrp,1/nrp,nrp)
        
        I = conc[t-1,conc[t-1,:] >= alpha2[t]] 
        II = np.where(conc[t-1,:] >= alpha2[t])[0]
        pr[t]=len(I)/nrp
        
        if len(I) < nrp:
            
            weight_risk = np.zeros((nrp,1))
            for p in range(len(II)):
                weight_risk[II[p],0] = 1/len(I)
            sr = systematic_resample(weight_risk)
        
        # MCMC iterations for each particle
        if np.mean(count[:,t-1]) < 0.3 and t > 1:
            jr = jr/1.1
        if np.mean(count[:,t-1]) > 0.3 and t > 1:
            jr = jr*1.1
            
        xx = Parallel(n_jobs=nrp)(delayed(mcmc_steps)(sigma[t-1,sr[p],:], jr) for p in range(nrp))
        
        for p in range(nrp):
            sigma[t,p,:] = xx[p][0]
            conc[t,p] = risk(xx[p][2])  
            ld_curr[p,:] = xx[p][2]
                    
            count[p,t] = np.sum(xx[p][4])/mc_steps
                    
        if alpha1[t] == 1:
            risk_ind = risk_ind + 1
            alpha2[t+1] = alpha2_fix[risk_ind]
                          
        t = t + 1
            
        if alpha2[t] >= threshold1 and alpha2[t-1] < threshold1:
            alpha2[t] = threshold1
            I = conc[t-1,conc[t-1,:] >= threshold1] 
            probb1[ppp] = len(I)/nrp*np.prod(pr[0:t])

        if alpha2[t] >= threshold2 and alpha2[t-1] < threshold2:
            alpha2[t] = threshold2
            I = conc[t-1,conc[t-1,:] >= threshold2] 
            probb2[ppp] = len(I)/nrp*np.prod(pr[0:t])
                    
    mdic1 = {'probb1':probb1, 'probb2':probb2, 'alpha1': alpha1, 'alpha2': alpha2, 'sigma': sigma, 'conc': conc, 'pr': pr}
    name1 = 'Diff_SMC2_fix_N.mat'
    scipy.savemat(name1, mdic1)

 