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
from generate_data import *

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
y_true = tl['data'].T
y_pure = tl['data1']
norm_true = tl['norm_ld']
ld_true = tl['ld_true']

#%%

inv = 'yes'

itt = 1000001
jr = np.linspace(0.006,0.1,10)
nr_chains = 1

posterior = np.zeros((nr_chains,itt,12)); count=0; 
R = np.zeros((nr_chains,itt))
for ch in range(nr_chains): 
    
    count = np.zeros((itt,1))
    # initialization
    
    norm_old = np.random.normal(size=nr_red)
    summa = np.zeros((nr_red,nr_space))
    for n in range(nr_red):
        summa[n,:] = norm_old[n]*np.sqrt(ww[n])*vv[:,n]
             
    ld_old = mean_ld + np.sum(summa, 0)
    
    if inv == 'yes': 
        y_old = forward(ld_old, s)     
        lh_old = np.sum(stats.norm.logpdf(y_old, y_true, obs_noise))
                        
    if inv == 'no':
        lh_old = 1
        
    prior_old = np.prod(stats.norm.pdf(norm_old))
    risk_old = risk(ld_old)

    for it in range(itt):
        
        if it > 1001 and np.mod(it,200)==1 and np.sum(count[it-201:it-1])/200 < 0.3:
            jr = jr/1.1
        if it > 1001 and np.mod(it,200)==1 and np.sum(count[it-201:it-1])/200 > 0.3:
            jr = jr*1.1
        
        norm_new = np.zeros((10,1))
        for com in range(10):
            norm_new[com] = norm_old[com] + np.random.normal(0,jr[com],1) # np.sqrt(1-jr**2)*norm_old + np.random.normal(0,jr,10)
        norm_new = norm_new.flatten()

        summa = np.zeros((nr_red,nr_space))
        for n in range(nr_red):
            summa[n,:] = norm_new[n]*np.sqrt(ww[n])*vv[:,n]
                 
        ld_new = mean_ld + np.sum(summa, 0)
        
        if inv == 'yes': 
            y_new = forward(ld_new, s)     
            lh_new = np.sum(stats.norm.logpdf(y_new, y_true, obs_noise))
                            
        if inv == 'no':
            lh_new = 1
        
        prior_new = np.prod(stats.norm.pdf(norm_new))
        
        acc = min(1,np.exp(lh_new-lh_old)*prior_new/prior_old)
        
        if np.random.uniform(0,1,1) < acc:
            count[it] = 1
            norm_old = norm_new
            ld_old = ld_new
            lh_old = lh_new; 
            prior_old = prior_new
            risk_old = risk(ld_new)
        
        posterior[ch,it,0:10] = norm_old
        posterior[ch,it,10] = lh_old
        posterior[ch,it,11] = prior_old
        
        # risk
        R[ch, it] = risk_old

        if np.mod(it,10000)==0:
            mdic1 = {'posterior':posterior, 'count':count, 'it': it, 'R':R}
            name1 = 'Diff_MCMC1.mat'
            scipy.savemat(name1, mdic1)
