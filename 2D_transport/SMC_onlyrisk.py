# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:01:13 2022

@author: lfriedl2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.io as scipy
import flopy
import flopy.modflow as mf
import flopy.mt3d as mt
import flopy.utils as fu
import math
from scipy import special
import flopy.utils.binaryfile as bf
from joblib import Parallel, delayed
from numpy.random import random
from multiprocessing import Process
import pathlib


tl = scipy.loadmat('Data_occ1.mat')
y_true = tl['data'].flatten()
y_pure = tl['data0'].flatten()
nr_data = np.int(tl['nr_data'].flatten())
obs_noise = tl['obs_noise'].flatten()
cm_chol = tl['cm_chol']
mean = tl['mean'].flatten()
R_true = tl['R_true'].flatten()
Yfield_true = tl['Yfield']
pumping_rate = tl['pumping_rate'].flatten()
source_conc = tl['source_conc'].flatten()

h_diff = tl['h_diff'].flatten()
outlet1 = np.int(tl['outlet1'].flatten())
outlet2 = np.int(tl['outlet2'].flatten())
poro = 0.3

#MODFLOW 2005
modelname = 'example'
mf_model = mf.Modflow(modelname = modelname, exe_name='mf2005')
#DIS file
Lx = 250 #[m]
Ly = 250 #[m]
nrow = 51
ncol = 51
nlay = 1
delr = Lx / ncol
delc = Ly / nrow
H = Lx/nrow # acquifer thickness
top = H*np.ones((nrow, ncol))
botm = np.zeros((nrow, ncol))


#%%

# surpress the console output
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def data_forward(Yf, p):
    
    modelname = "example_{}".format(p)
    mf_model = mf.Modflow(modelname = modelname, exe_name='mf2005')
    
    
    # pumping well
    x_pump = 25
    y_pump = 25

    t = np.zeros((1,nrow,ncol))
    t[0,:,:] = np.exp(Yf)

    # initial and boundary conditions
    h = 2.5  # initial constant head left [m]

    dis = mf.ModflowDis(mf_model, nlay, nrow, ncol, delr = delr, delc = delc, 
                        top = top, botm = botm, laycbd = 0, itmuni=1, 
                        nstp = 1, steady=True)

    # Output Control: Create a flopy output control object
    oc = mf.ModflowOc(mf_model)

    #BCF file
    bcf = flopy.modflow.mfbcf.ModflowBcf(mf_model,laycon=0, tran=t) #confined

    #BAS file
    ibound = np.ones((nlay, nrow, ncol)) #active
    # ibound[0, 0, :] = -1 # constant head
    # ibound[0, -1, :] = -1
    ibound[0, :, 0] = -1  # left of domain
    ibound[0, :, -1] = -1  # right of domain

    strt = np.ones((nlay, nrow, ncol), dtype=np.float32)
    strt[:, :, 0] = h
    strt[:, :, -1] = 0.0

    bas = mf.ModflowBas(mf_model, ibound = ibound, strt = strt)

    #PCG file
    pcg = flopy.modflow.mfpcg.ModflowPcg(mf_model, mxiter=20, iter1=30, hclose=1e-03, rclose=1e-03, relax=1.0)

    #CHD
    #[lay, row, col, shead, ehead
    chd= h
    chd_data = []
    for c in range(nrow):
        dd = np.array([0, c, nrow-1, 0, 0])
        chd_data.append(dd)
    for c in range(1,nrow):
        dd = np.array([0, c, 0, chd, chd])
        chd_data.append(dd)
    stress_period_data = {0:chd_data}
    stress_period_data
    chd = mf.mfchd.ModflowChd(mf_model, stress_period_data=stress_period_data)

    #WELL
    #[lay, row, col, pumping rate]
    wel_sp1 = [[0, x_pump, y_pump, pumping_rate]]
    stress_period_data = {0: wel_sp1}
    wel = flopy.modflow.ModflowWel(mf_model, stress_period_data=stress_period_data)

    #LMT Linkage with MT3DMS for multi-species mass transport modeling
    # lmt = flopy.modflow.ModflowLmt(mf_model, output_file_name='mt3d_link.ftl')
    #Write input files
    success = False
    while success == False:
        try:
            mf_model.write_input()
        except:
            success = False
        else:
            success = True

    # run the model
    success = False
    while success == False:
        try:
            with suppress_stdout():
                mf_model.run_model()
        except:
            success = False
        else:
            success = True
    
    headobj = bf.HeadFile(modelname+'.hds')
    head = headobj.get_data(totim=1.0)

    if nr_data == 8:
        data0 = np.zeros((8,1))
        data0[0]= head[0, 12, 12].flatten()
        data0[1]= head[0, 12, 25].flatten()
        data0[2] = head[0, 12, 38].flatten()
        data0[3] = head[0, 25, 12].flatten()
        data0[4] = head[0, 25, 38].flatten()
        data0[5] = head[0, 38, 12].flatten()
        data0[6] = head[0, 38, 25].flatten()
        data0[7] = head[0, 38, 38].flatten()
        
    if nr_data == 4:
        data0 = np.zeros((4,1))
        data0[0]= head[0, 12, 12].flatten()
        data0[1] = head[0, 12, 38].flatten()
        data0[2] = head[0, 38, 12].flatten()
        data0[3] = head[0, 38, 38].flatten()
    
    return data0
    
def risk_forward(Yf, p, perlen):
    
    new_folder_name = "Particle_{}".format(p)
    if not os.path.exists(new_folder_name):
    # Create the new folder
        os.mkdir(new_folder_name)
    os.chdir(new_folder_name)
    
    t = np.zeros((1,nrow,ncol))
    t[0,:,:] = np.exp(Yf)
    
    x_conc = np.linspace(0,nrow-1,nrow,dtype=int)
    y_conc = 0
    initial_conc = 0

    # initial and boundary conditions
    h = 0  # initial constant head everywhere [m]
    h_diff = 2.5
    h_l = h + h_diff  # constant head on the left of model domain
    h_r = h # constant head on the right of model domain

    modelname1 = "example1_{}".format(p)
    mf_model1 = mf.Modflow(modelname = modelname1, exe_name='mf2005.exe')

    dis = mf.ModflowDis(mf_model1, nlay, nrow, ncol, delr = delr, delc = delc, 
                        top = top, botm = botm, laycbd = 0, itmuni=1, perlen = perlen, 
                        nstp = 1, steady=False)

    # Output Control: Create a flopy output control object
    # Output Control: Create a flopy output control object
    oc = mf.ModflowOc(mf_model1)

    #BCF file
    bcf = flopy.modflow.mfbcf.ModflowBcf(mf_model1, laycon=0, tran=t) #confined

    #BAS file
    ibound = np.ones((nlay, nrow, ncol)) #active
    ibound[0, :, 0] = -1 
    ibound[0, :, -1] = -1
    # ibound[0, 0, :] = 0 
    # ibound[0, -1, :] = 0

    strt = h #starting head

    bas = mf.ModflowBas(mf_model1, ibound = ibound, strt = strt)

    #PCG file
    pcg = flopy.modflow.mfpcg.ModflowPcg(mf_model1, mxiter=20, iter1=30, hclose=1e-03, rclose=1e-03, relax=1.0)

    #[lay, row, col, shead, ehead

    chd_data = []
    for row_col in range(0, ncol):
        chd_data.append([0, row_col, 0, h_l, h_l])
        chd_data.append([0, row_col, nrow - 1, h_r, h_r])
        # if row_col != 0 and row_col != N - 1:
        #     chd_rec.append(((layer, 0, row_col), -1))
        #     chd_rec.append(((layer, N - 1, row_col), -1))
    stress_period_data = {0:chd_data}
    stress_period_data
    chd = mf.mfchd.ModflowChd(mf_model1, stress_period_data=stress_period_data)

    #LMT Linkage with MT3DMS for multi-species mass transport modeling
    nameLMT = 'mt3d_link.ftl'
    lmt = flopy.modflow.ModflowLmt(mf_model1, output_file_name=nameLMT)
    #Write input files
    success = False
    while success == False:
        try:
            mf_model1.write_input()
        except:
            success = False
        else:
            success = True

    #Run the model
    success = False
    while success == False:
        try:
            with suppress_stdout():
                mf_model1.run_model()
        except:
            success = False
        else:
            success = True
            
    headobj = bf.HeadFile(modelname1+'.hds')
    head = headobj.get_data(totim=perlen)
    
    ##########################################################
    #MT3D-USGS
    namemt3d="modelnamemt3d_{}".format(p)
    mt_model = mt.Mt3dms(modelname=namemt3d, version='mt3d-usgs', modflowmodel=mf_model1)

    #BTN file
    icbund = np.ones((nlay, nrow, ncol))
    icbund[0, x_conc, y_conc] = -1 # constant concentration
    btn = flopy.mt3d.Mt3dBtn(mt_model, sconc=initial_conc, prsity=poro, munit='g', icbund=icbund)

    #ADV file
    mixelm = -1 #Third-order TVD scheme (ULTIMATE)
    percel = 1 #Courant number PERCEL is also a stability constraint
    adv = flopy.mt3d.Mt3dAdv(mt_model, mixelm=mixelm, percel=percel)

    #GCG file
    mxiter = 1 #Maximum number of outer iterations
    iter1 = 200 #Maximum number of inner iterations
    isolve = 3 #Preconditioner = Modified Incomplete Cholesky
    gcg = flopy.mt3d.Mt3dGcg(mt_model, mxiter=mxiter, iter1=iter1, isolve=isolve)

    #DSP file
    al = 1 #longitudinal dispersivity
    dmcoef = 1e-09 #effective molecular diffusion coefficient (salt)
    trpt = .1 #ratio of the horizontal transverse dispersivity to the longitudinal dispersivity
    trpv = .1 #ratio of the vertical transverse dispersivity to the longitudinal dispersivity
    dsp = mt.Mt3dDsp(mt_model, al=al, dmcoef=dmcoef, trpt=trpt, trpv=trpv)

    #SSM file
    itype = flopy.mt3d.Mt3dSsm.itype_dict()
    #[K,I,J,CSS,iSSType] = layer, row, column, source concentration, type of sink/source: well-constant concentration cell 
    ssm_data = []
    ssm_data = [[0, x_conc[0], y_conc, source_conc, 2]]  # 2 is a well
    ssm_data.append([0, x_conc[0], y_conc, source_conc, -1])  # constant concentration cell
    for cc in range(len(x_conc)+1):
        ssm_data.append([0, x_conc[cc-1], y_conc, source_conc, 2])  # 2 is a well
        ssm_data.append([0, x_conc[cc-1], y_conc, source_conc, -1])  # constant concentration cell
    ssm_data = {0:ssm_data}
    ssm = flopy.mt3d.Mt3dSsm(mt_model, stress_period_data=ssm_data)

    #Write model input
    success = False
    while success == False:
        try:
            mt_model.write_input()
        except:
            success = False
        else:
            success = True

    #Run the model
    success = False
    tttry = 0
    while success == False and tttry < 5:
        try:
            with suppress_stdout():
                mt_model.run_model()
        except:
            success = False
            tttry = tttry + 1
        else:
            success = True
            
    if success == True:
        conc = fu.UcnFile("MT3D001.UCN")  # _{}".format(p)
        concc = conc.get_alldata()
        R = max(concc[0,0,outlet1:outlet2,49])
    if success == False:
        R = 0
        
    os.chdir("..")
    
    return R
   


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

def mcmc_steps(sigma, sr, jr, perlen, conc):
    
    normal_old = np.zeros((nrp,nrow*ncol))
    risk_old = np.zeros((nrp,1))
    for p in range(nrp):
        normal_old[p,:] = sigma[sr[p],:]
        risk_old[p] = conc[sr[p]]
    
    
    # xx = Parallel(n_jobs=nrp)(delayed(risk_forward)(Yf_old[p,:,:], p, perlen) for p in range(nrp))
    # risk_old = np.array(xx)
    
    lh_old = np.zeros((nrp))      
    arr = np.zeros((nrp, mc_steps))
    
    for it in range(mc_steps):
        
        normal_new = np.zeros((nrp,nrow*ncol))
        Yf_new = np.zeros((nrp,nrow,ncol))
        risk_new = np.zeros((nrp))
        for p in range(nrp):
            
            normal_new[p,:] = np.sqrt(1-jr**2)*normal_old[p,:] + jr*np.random.normal(0,1,ncol*nrow)
            Yf_new[p,:,:] = np.array(np.reshape(mean + np.dot(cm_chol, normal_new[p,:]).flatten(),(nrow,ncol)))
        
        risk_new= Parallel(n_jobs=nrp)(delayed(risk_forward)(Yf_new[p,:,:], p, perlen) for p in range(nrp))

        lh_new = np.zeros((nrp))
        
        aa = np.linspace(1,1,nrp) # np.exp((lh_new-lh_old)*alpha1[t])

        for p in range(nrp):
            if np.random.uniform(0,1,1) < aa[p] and risk_new[p] >= bt:
                arr[p,it] = 1
                normal_old[p,:] = normal_new[p,:]
                lh_old[p] = lh_new[p]
                risk_old[p] = risk_new[p]
            
    return normal_old, lh_old, risk_old, arr

#%% 

nrp = 40 # nr of particles
mc_steps = 10 # MH steps 
nr_run = 10
threshold1 = 10000 
threshold2 = 60 # final target threshold
bt = 1

# fixed sequence of thresholds
nr_risk = 30 # number of steps
x = np.linspace(0,nr_risk,nr_risk, dtype=int)
b=3500
a = (100-b)/np.log(nr_risk-1)
plt.plot(a*np.log(x)+b)
alpha2_fix = np.ceil(a*np.log(x)+b)

# simulation works more consistently for 'regular' numbers of days
for t in range(nr_risk):
    if alpha2_fix[t] > 1000:
        alpha2_fix[t] = alpha2_fix[t] - np.mod(alpha2_fix[t],50)
    if alpha2_fix[t] <= 1000 and alpha2_fix[t] > 100:
        alpha2_fix[t] = alpha2_fix[t] - np.mod(alpha2_fix[t],10)
    if alpha2_fix[t] <= 100:
        alpha2_fix[t] = alpha2_fix[t] - np.mod(alpha2_fix[t],5)

alpha2_fix[nr_risk-1] = 100
alpha2_fix = np.concatenate([alpha2_fix, np.linspace(95,60,8)])

plt.plot(alpha2_fix)

probb1 = np.zeros((nr_run))
probb2 = np.zeros((nr_run))

nr = np.zeros((nr_run))

for ppp in range(nr_run):
    
    ittotal = 1000
    alpha1 = np.ones((ittotal)) 
    alpha2 = np.zeros((ittotal))  + threshold1
    jr = 1.0
    weight = np.zeros((ittotal,nrp))
    weight_norm = np.zeros((ittotal,nrp))
    sigma = np.zeros((ittotal,nrp,nrow*ncol))
    conc = np.zeros((ittotal,nrp))
    pr = np.ones((ittotal,1))
    count = np.zeros((nrp,ittotal))
    acc_R = np.zeros((ittotal, nrp))

    nr_resample = 0
    
    # initialization
    sigma[0,:,:] = stats.norm.rvs(size=(nrp,nrow*ncol))
    weight[0,:] = np.linspace(1/nrp,1/nrp,nrp)
    weight_norm[0,:] = np.linspace(1/nrp,1/nrp,nrp)
    
    alpha2[1] = alpha2_fix[1]
    t = 1
    risk_ind = 0
    while alpha2[t]>threshold2:
        
        Yfield_curr = np.zeros((nrp, nrow*ncol))
        for p in range(nrp): 
            Yfield_curr[p,:] = mean + np.dot(cm_chol, sigma[t-1,p,:].flatten())
            
        conc[t,:] = Parallel(n_jobs=nrp)(delayed(risk_forward)(np.reshape(Yfield_curr[p,:],(nrow,ncol)), p, alpha2[t]*24*3600) for p in range(nrp))

        sr = range(nrp)        
        I = conc[t,conc[t,:] >= bt] 
        II = np.where(conc[t,:] >= bt)[0]
        pr[t]=len(I)/nrp
                
        
        if len(I) < nrp:
            
            weight_risk = np.zeros((nrp,1))
            for p in range(len(II)):
                weight_risk[II[p],0] = 1/len(I)
            sr = systematic_resample(weight_risk)
            
        xx = mcmc_steps(sigma[t-1,:,:],sr, jr, alpha2[t]*24*3600, conc[t,:])

        sigma[t,:,:] = xx[0]
        conc[t,:] = xx[2].flatten()
        
        for p in range(nrp):
            count[p,t] = np.sum(xx[3][p,:])/mc_steps
            
        t = t + 1
        alpha2[t] = alpha2_fix[t]
        
        jr = jr/1.05
        
        plt.plot(conc[0:t,:])
        plt.show()
        
        if alpha2[t] <= threshold2:
            alpha2[t] = threshold2
            
            Yfield_curr = np.zeros((nrp, nrow*ncol))
            for p in range(nrp): 
                Yfield_curr[p,:] = mean + np.dot(cm_chol, sigma[t-1,p,:].flatten())
                conc[t,p] = risk_forward(np.reshape(Yfield_curr[p,:],(nrow,ncol)), p, alpha2[t]*24*3600)
               
            I = conc[t,conc[t,:] >= bt] 
            II = np.where(conc[t,:] >= bt)[0]
            probb2[ppp] = len(I)/nrp*np.prod(pr[0:t])

    mdic1 = {'pr': pr, 'probb1':probb1, 'probb2':probb2, 'alpha1': alpha1, 'alpha2': alpha2, 'sigma': sigma[0,:,:],'sigmaa': sigma[t-1,:,:],'sigmaaa': sigma[t,:,:], 'conc': conc, 'weight_norm': weight_norm, 'nr_resample':nr_resample, 'II':II }
    name1 = 'Flow_priorrisk_{}.mat'.format(ppp)
    scipy.savemat(name1, mdic1)
 