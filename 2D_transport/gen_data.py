# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:09:18 2023

@author: lfriedl2
"""

#Import
import numpy as np
import matplotlib.pyplot as plt
import flopy
import flopy.modflow as mf
import flopy.mt3d as mt
import flopy.utils as fu
import math
from scipy import special
from scipy.linalg import toeplitz 
import flopy.utils.binaryfile as bf

def CF(sh,Lag,cl,H):
    CM = sh**2*np.exp(-np.power((Lag/(cl)),(2*H)));
    return CM
  
def block_toeplitz(c,d):

    cmm = np.zeros((d*d,d*d)); cmmm = np.zeros((d*d,d*d))
    
    for d1 in range(d): # columns of block
        p3 = d1*d; p4 = (d1+1)*d;
        cc = np.roll(c,d*d1)                  
            
        for d2 in range(d):
                p1 = d2*d; p2 = (d2+1)*d;
                cmm[p1:p2,p3:p4] = toeplitz(cc[p1:p2])
      
    for i in range(d*d):
        cmmm[i,i]=cmm[i,i];
        for j in range(i):          
            cmmm[i,j]=cmm[i,j]  
            cmmm[j,i]=cmm[i,j] 
            
    return cmmm
                

#%%%%%

#MODFLOW 2005
modelname = 'example'
mf_model = mf.Modflow(modelname = modelname, exe_name='mf2005.exe')
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

# transmissivity field  [m2 / s]
mean=np.log(5*10**(-5))
sh=3;
cl= 25/delr; #in nr cells --> *Lx/nrow gives meters
angle_aniso=90;
ratio=1;
H=0.5

cl*Lx/nrow

cm=np.zeros((int(nrow)*int(ncol))); l=np.zeros((int(nrow)*int(ncol)));
Lag = np.zeros((nrow,ncol))
    # x is horizontal, y is vertical, change first x (rowwise)
for s in range(int(nrow)):   #y first
    for t in range(int(ncol)):  #x first
            Lag[s,t] =  np.sqrt(1/ratio**2*(t-0)**2 + 1/1**1*(s-0)**2)
            px = (s)*(int(nrow))+(t+1);
            l[px-1]=Lag[s,t]; 
            cm[px-1]=CF(sh,Lag[s,t],cl,H);   
cm = block_toeplitz(cm,nrow)
cm_chol = np.linalg.cholesky(cm)

nr_data = 4
outlet1 = 23
outlet2= 28

#%%

z = np.random.normal(0,1,(nrow*ncol,1))
Yfield = np.reshape(mean + np.dot(cm_chol, z).flatten(),(nrow,ncol))

fig = plt.gcf()
plt.rcParams.update({'font.size': 15})
plt.subplots_adjust(left=0.2,
                    bottom=0.2, 
                    right=0.8, 
                    top=0.9, 
                    wspace=0.35, 
                    hspace=0.2)
plt.imshow(Yfield)
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.set_label('Log-transmissivity [-]', rotation=270, labelpad = 15)
ax = plt.gca()
plt.clim(-20,0)
plt.xticks([0,25,50], ['0','125', '250'])
plt.yticks([0,25,50], ['250', '125','0'])
plt.xlabel('x [m]')
plt.ylabel('y [m]')
fig.set_size_inches(6,6)
fig.savefig('transm_field.png', dpi=100)


#%%

# pumping well
pumping_rate = -0.0005 #m3/s
x_pump = 25
y_pump = 25

t = np.zeros((1,nrow,ncol))
t[0,:,:] = np.exp(Yfield)

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
mf_model.write_input()

# run the model
mf_model.run_model()

import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf

# Create the headfile object
headobj = bf.HeadFile(modelname+'.hds')
head = headobj.get_data(totim=1.0)
times = headobj.get_times()

# Setup contour parameters
levels = np.arange(-100,2.5, 0.5)
extent = (delr/2., Lx - delr/2., delc/2., Ly - delc/2.)

# Make the plots
fig = plt.gcf()
plt.rcParams.update({'font.size': 15})
plt.subplots_adjust(left=0.2,
                    bottom=0.2, 
                    right=0.8, 
                    top=0.9, 
                    wspace=0.35, 
                    hspace=0.2)
plt.imshow(head[0, :, :], extent=extent, cmap='YlGnBu', vmin=-100, vmax=2.5)
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.set_label('Head [m]', rotation=270, labelpad = 15)
ax = plt.gca()
plt.clim(0,2.5)
contours = plt.contour(np.flipud(head[0, :, :]), levels=levels, extent=extent, zorder=10)
plt.clabel(contours)
plt.clabel(contours, inline=1, fontsize=10, fmt='%d', zorder=11)
plt.scatter(Lx/2, Ly/2, s=50, c='red', marker='o')
plt.scatter(3*Lx/4, Ly/4, s=50, c='red', marker='o')
plt.scatter(Lx/4, 3*Ly/4, s=50, c='red', marker='o')
plt.scatter(3*Lx/4, 3*Ly/4, s=50, c='red', marker='o')
plt.scatter(Lx/4, Ly/4, s=50, c='red', marker='o')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xticks([0,125,250], ['0','125', '250'])
plt.yticks([0,125,250], ['0', '125','250'])
fig.set_size_inches(6,6)
fig.savefig('head_field.png', dpi=100)


obs_noise = 0.02

if nr_data == 8:
    data0 = np.zeros((8,1))
    data0[0]= head[0, 12, 12]
    data0[1]= head[0, 12, 25]
    data0[2] = head[0, 12, 38]
    data0[3] = head[0, 25, 12]
    data0[4] = head[0, 25, 38]
    data0[5] = head[0, 38, 12]
    data0[6] = head[0, 38, 25]
    data0[7] = head[0, 38, 38]
    data = data0 + np.random.normal(0,obs_noise,(8,1))
    
if nr_data == 4:
    data0 = np.zeros((4,1))
    data0[0]= head[0, 12, 12]
    data0[1] = head[0, 12, 38]
    data0[2] = head[0, 38, 12]
    data0[3] = head[0, 38, 38]
    data = data0 + np.random.normal(0,obs_noise,(4,1))


#%% TRANSPORT

# contamination source
x_conc = np.linspace(0,nrow-1,nrow,dtype=int)
y_conc = 0
poro = 0.3
initial_conc = 0
source_conc = 1000 # mg
perlen = 60*24*3600  # in seconds

# initial and boundary conditions
h = 0  # initial constant head everywhere [m]
h_diff = 2.5
h_l = h + h_diff  # constant head on the left of model domain
h_r = h # constant head on the right of model domain

modelname1 = 'example1'
mf_model1 = mf.Modflow(modelname = modelname1, exe_name='mf2005.exe')

dis = mf.ModflowDis(mf_model1, nlay, nrow, ncol, delr = delr, delc = delc, 
                    top = top, botm = botm, laycbd = 0, itmuni=1, perlen = perlen, 
                    nstp = 1, steady=False)

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
mf_model1.run_model()

# Create the headfile object
headobj = bf.HeadFile(modelname1+'.hds')
head = headobj.get_data(totim=perlen)
times = headobj.get_times()

# Setup contour parameters
levels = np.arange(0, h_l, 5)
extent = (delr/2., Lx - delr/2., delc/2., Ly - delc/2.)

# Make the plots
plt.subplot(1, 1, 1, aspect='equal')
plt.title('Head distribution (m)')
plt.imshow(head[0, :, :], extent=extent, cmap='YlGnBu', vmin=0., vmax=h_l)
plt.colorbar()
plt.show()


##########################################################
#MT3D-USGS
namemt3d='modelnamemt3d'
mt_model = mt.Mt3dms(modelname=namemt3d, version='mt3d-usgs', modflowmodel=mf_model1)

#BTN file
icbund = np.ones((nlay, nrow, ncol))
icbund[0, x_conc, y_conc] = -1 # constant concentration
btn = flopy.mt3d.Mt3dBtn(mt_model, sconc=initial_conc, prsity=poro, munit='mg', icbund=icbund, tunit="S", nprs=0)

#ADV file
mixelm = -1 #Third-order TVD scheme (ULTIMATE)
percel = 1 #Courant number PERCEL is also a stability constraint
adv = flopy.mt3d.Mt3dAdv(mt_model, mixelm=mixelm, percel=percel)

#GCG file
mxiter = 1 #Maximum number of outer iterations
iter1 = 50 #Maximum number of inner iterations
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
mt_model.run_model()

#Plot concentration results
conc = fu.UcnFile('MT3D001.UCN')
concc = conc.get_alldata()

fig = plt.gcf()
plt.rcParams.update({'font.size': 15})
plt.subplots_adjust(left=0.2,
                    bottom=0.2, 
                    right=0.85, 
                    top=0.9, 
                    wspace=0.35, 
                    hspace=0.2)
plt.imshow(concc[0,0, :, :], extent=extent)
plt.colorbar(label = 'Concentration [mg/l]', fraction=0.046, pad=0.04)
plt.clim(0,1)
plt.vlines(51*delr, outlet1*5, outlet2*5, colors='red', linewidth=10)
plt.title('')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
fig.set_size_inches(6,6)
fig.savefig('conc_field.png', dpi=100)

#%%

R_true = max(concc[0,0,outlet1:outlet2,49]>=1)

fig = plt.gcf()
plt.rcParams.update({'font.size': 15})
plt.subplots_adjust(left=0.2,
                    bottom=0.2, 
                    right=0.85, 
                    top=0.9, 
                    wspace=0.35, 
                    hspace=0.2)
plt.imshow(concc[0,0, :, :]/1000, extent=extent)
plt.colorbar(label = 'Concentration [g/l]', fraction=0.046, pad=0.04)
plt.clim(0,1)
plt.vlines(50*5, outlet1*5, outlet2*5, colors='red', linewidth=10)
plt.title('')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
fig.set_size_inches(6,6)
fig.savefig('conc_field1.png', dpi=100)


#%% save data
import scipy.io as scipy
mdic1 = {'data': data, 'data0': data0, 'obs_noise': obs_noise, 
         'Lx':Lx, 'Ly': Ly,
         'nrow': nrow, 'ncol':ncol, 'nlay': nlay,
         'mean': mean, 'sh': sh, 'cl': cl, 'ratio': ratio, 'H': H,
         'cm_chol': cm_chol, 'Yfield':Yfield,'outlet1':outlet1, 'outlet2':outlet2, 
         'pumping_rate': pumping_rate, 'h':h, 'nr_data': nr_data,
         'source_conc': source_conc, 'h_diff': h_diff, 'perlen': perlen, 'R_true': R_true}
name1 = 'Data_occ1.mat'
scipy.savemat(name1, mdic1)



#%% local conditioning

import scipy

Yfield_true = Yfield

vec = [25*51+25, 12*51+12, 12*51+38,  38*51+12,  38*51+38]

# local measurements
field_c = np.zeros((5,1))
field_c[0] = Yfield_true[25,25]
field_c[1] = Yfield_true[12,12]
field_c[2] = Yfield_true[12,38]
field_c[3] = Yfield_true[38,12]
field_c[4] = Yfield_true[38,38]

lobs_noise = 0.1
cm_lobs = lobs_noise**2*np.eye(5)

J = np.zeros((5,51*51))
J[0,vec[0]] = 1
J[1,vec[1]] = 1
J[2,vec[2]] = 1
J[3,vec[3]] = 1
J[4,vec[4]] = 1

cm_loc = np.linalg.inv(np.linalg.inv(cm)+np.dot(J.T,np.dot(np.linalg.inv(cm_lobs),J)))
cm_loc_chol = scipy.linalg.cholesky(cm_loc, lower = True, overwrite_a=True ) ## artefact???

mean_loc = np.dot(cm_loc,(np.dot(J.T,np.dot(np.linalg.inv(cm_lobs),field_c)) + np.dot(np.linalg.inv(cm),mean*np.ones((51*51,1)))))

plt.imshow(np.reshape(mean_loc,(51,51)))
plt.colorbar()
plt.clim(-20,0)
plt.show()

plt.imshow(np.reshape(np.diag(cm_loc),(51,51)))
plt.colorbar()
plt.show()

plt.imshow(np.reshape(mean_loc.flatten() + np.dot(cm_loc_chol, np.random.normal(size=(1,51*51)).flatten()),(51,51)))
plt.colorbar()
plt.clim(-20,0)
plt.show()

mdic1 = {'mean_loc': mean_loc, 'cm_loc':cm_loc, 'cm_loc_chol':cm_loc_chol}
name1 = 'krig1.mat'
scipy.io.savemat(name1, mdic1)








