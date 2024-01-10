# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:01:13 2022

@author: lfriedl2
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy.io 
import flopy
import flopy.modflow as mf
import flopy.mt3d as mt
import flopy.utils as fu
import math
from scipy import special
import flopy.utils.binaryfile as bf
from joblib import Parallel, delayed
from numpy.random import random
from scipy.linalg import toeplitz 

tl = scipy.io.loadmat('Data_occ1.mat')
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
perlen = tl['perlen'].flatten()
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

tl = scipy.io.loadmat('krig1.mat')
mean_loc = tl['mean_loc'].flatten()
cm_loc = tl['cm_loc']
cm_loc_chol = tl['cm_loc_chol']



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
                        top = top, botm = botm, laycbd = 0, itmuni=4, 
                        nstp = 1)

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
        data0[4] = head[0, 25, 39].flatten()
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
    
def risk_forward(Yf, p):
    
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
                        top = top, botm = botm, laycbd = 0, itmuni=4, perlen = perlen, 
                        nstp = 1)

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

    #CHD
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
    lmt = flopy.modflow.ModflowLmt(mf_model1, output_file_name='mt3d_link.ftl')
    #Write input files
    mf_model1.write_input()

    # run the model
    try: 
        with suppress_stdout():
            mf_model1.run_model()
    except:
        head = np.zeros((1,nrow,ncol))
    else:
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
    mt_model.write_input()

    #Run the model
    with suppress_stdout():
        mt_model.run_model()
    
    conc = fu.UcnFile('MT3D001.UCN')
    concc = conc.get_alldata()
    R = max(concc[0,0,outlet1:outlet2,49])

    
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



#%%%%%%

jr = 1.0
mc_steps = 40*48000
nr_risk = 38*10*40 
thin = np.ceil(mc_steps/nr_risk)
sigma = np.zeros((nr_risk, 51*51))
lh = np.zeros((nr_risk, 1))

normal_old = stats.norm.rvs(size=(51*51,1))
Yf_old = np.array(np.reshape(mean_loc.flatten() + np.dot(cm_loc_chol, normal_old).flatten(),(nrow,ncol)))
y_old = data_forward(Yf_old, 0)         
lh_old = np.sum(stats.norm.logpdf(y_old.flatten(), y_true, obs_noise))  
    
arr = np.zeros((mc_steps,1))
cc = 0

mc_steps = np.int((15000-5500)*thin)

for it in range(mc_steps):
        
    normal_new = np.sqrt(1-jr**2)*normal_old.flatten() + jr*np.random.normal(0,1,nrow*ncol).flatten()
    Yf_new = np.array(np.reshape(mean_loc.flatten() + np.dot(cm_loc_chol, normal_new).flatten(),(nrow,ncol)))

    y_new = data_forward(Yf_new, 0)         
    lh_new = np.sum(stats.norm.logpdf(y_new.flatten(), y_true, obs_noise))
        
    aa = np.exp((lh_new-lh_old))
    if np.random.uniform(0,1,1) < aa:
        arr[it] = 1
        normal_old = normal_new
        lh_old = lh_new
        
    if np.mod(it,thin) == 0: 
        sigma[cc,:] = normal_old.flatten()
        lh[cc] = lh_old
        cc = cc + 1

    if np.sum(arr[it-100:it])/100 < 0.25 and it > 100 and jr > 0 and np.mod(it,100)==0:
        jr = max(jr/1.1,0)
    if np.sum(arr[it-100:it])/100 > 0.35 and it > 100 and jr < 1 and np.mod(it,100)==0:
        jr = min(jr*1.1,1)
                         
mdic1 = {'sigma':sigma}
name1 = 'Flow_posterior_MCMC.mat'
scipy.io.savemat(name1, mdic1)

