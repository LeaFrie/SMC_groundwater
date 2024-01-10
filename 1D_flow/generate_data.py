# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 08:46:39 2023

@author: lfriedl2
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scipy
import statistics as stats


def forward(ld_true):
    A = np.zeros((nr_space, nr_space))
    a = np.exp(ld_true)
    for r in range(nr_space):
        for c in range(nr_space):
            if r > 0 and r < (nr_space-1):
                a_minus = stats.harmonic_mean([a[r],a[r-1]])  
                a_plus = stats.harmonic_mean([a[r],a[r+1]])   
                if r == c:
                    A[r,c] = - 1/(l_grid/nr_space)**2*(a_plus + a_minus)
                if r + 1 == c:
                    A[r,c] = 1/(l_grid/nr_space)**2*(a_plus)
                if r - 1 == c:
                    A[r,c] = 1/(l_grid/nr_space)**2*(a_minus)
                    
    AA = np.zeros((nr_space+2, nr_space+2)); AA[1:-1, 1:-1] = A;
    
    AA[0,0] = 1; AA[-1,-1] = 1  
    
    AA[1,0] = 1/(l_grid/nr_space)**2*(2*a[0])
    AA[1,1] = - 1/(l_grid/nr_space)**2*(2*a[0] + stats.harmonic_mean([a[0],a[1]]))
    AA[1,2] = 1/(l_grid/nr_space)**2*stats.harmonic_mean([a[0],a[1]])  
     
    AA[-2,-3] = 1/(l_grid/nr_space)**2*stats.harmonic_mean([a[-1],a[-2]])  
    AA[-2,-2] = - 1/(l_grid/nr_space)**2*(stats.harmonic_mean([a[-1],a[-2]]) + 2*a[-1])
    AA[-2,-1] = 1/(l_grid/nr_space)**2*(2*a[-1]) 

    b = np.zeros((nr_space,1))
    b[np.argmin(np.abs(knots - 0.25))] = s
    b[np.argmin(np.abs(knots - 0.5))] = s
    b[np.argmin(np.abs(knots - 0.75))] = s
    source = np.zeros((nr_space+2,1));
    source[1:-1,0] = b.flatten()

    u = -np.dot(np.linalg.inv(AA),source)
                
    return(u)

def u_exact_stationary(x, a, u_L, u_R):
        """
        Return stationary solution of a 1D variable coefficient
        Laplace equation: (a(x)*v'(x))'=0, v(0)=u_L, v(L)=u_R.
        v(x) = u_L + (u_R-u_L)*(int_0^x 1/a(c)dc / int_0^L 1/a(c)dc)
        """
        Nx = x.size - 1
        g = np.zeros(Nx+1)    
        dx = x[1] - x[0]   
        i = 0
        g[i] = 0.5*dx/a[i]
        for i in range(1, Nx):
            g[i] = g[i-1] + dx/a[i]
        i = Nx
        g[i] = g[i-1] + dx/a[i]
        v = u_L + (u_R - u_L)*g/g[-1]
        return v

def risk(ldd):
    R = stats.hmean(np.exp(ldd))

    return R

def corrf(dist, lamb, sd):
    return(sd**2*np.exp(-dist/lamb))

#%% log-conductivity 1D: Gausssian random process  (KL-expansion)

nr_space = 40
l_grid = 1

mean_ld = np.log(10**(-5))
sd_ld = 3
lamb = 0.3
nr_red = 10

knots = np.linspace(0,1,nr_space+1)
cell_m = np.linspace(0,l_grid, nr_space)+l_grid/(2*nr_space)

covm = np.zeros((nr_space, nr_space))
for i in range(nr_space):
    for ii in range(nr_space):
        covm[i,ii] = corrf(np.abs(knots[i]-knots[ii]), lamb, sd_ld) 

w,v = np.linalg.eig(covm)

norm_ld = np.random.normal(size=nr_red)
summa = np.zeros((nr_red,nr_space))
for n in range(nr_red):
    summa[n,:] = norm_ld[n]*np.sqrt(w[n])*v[:,n]
         
ld_true = mean_ld + np.sum(summa, 0)

fig = plt.gcf()
fig.subplots_adjust(bottom=0.2, left=0.2)
plt.rcParams.update({'font.size': 20})
plt.plot(cell_m,ld_true, linewidth=3)
plt.ylim([-20,-4])
fig.set_size_inches(8,5)
plt.xlabel('x [m]') 
plt.ylabel('Log-conductivity [-]')
fig.savefig('log_cond.png', dpi=100)

#%% measurements

s = 0.001  #m^3/s
obs_noise=0.01

u_true = forward(ld_true)

fig = plt.gcf()
fig.subplots_adjust(bottom=0.2, left=0.2)
plt.rcParams.update({'font.size': 20}) 
plt.plot(cell_m, u_true[1:-1], linewidth=3)
plt.vlines(0.25+l_grid/(2*nr_space),0,max(u_true), linestyles='--', linewidth=3, color='black')
plt.vlines(0.5+l_grid/(2*nr_space),0,max(u_true), linestyles='--', linewidth=3, color='black')
plt.vlines(0.75+l_grid/(2*nr_space),0,max(u_true), linestyles='--', linewidth=3, color='black')    # peaks after the location of the sources (solving from left to right)


dd_sens = np.linspace(0,1,13)
d_sens  = dd_sens[1]-dd_sens[0]
x_sens = np.linspace(d_sens,1-d_sens,11)

uu_true = u_true[1:-1]
data1 = np.zeros((len(x_sens),1))
for ind in range(len(x_sens)):
    data1[ind,0] = uu_true[np.argmin(np.abs(knots - x_sens[ind]))]


error = np.random.normal(0,obs_noise,size = 11)   
data = data1.flatten() + error

plt.scatter(x_sens + l_grid/(2*nr_space) ,data, marker='X', color='red', linewidths=0.5, s=80)
fig.set_size_inches(8,5)
plt.xlabel('x [m]') 
plt.ylabel('Hydraulic head h [m]')
fig.savefig('log_cond_data.png', dpi=100)

#%% risk: flow from left to right

R_true = risk(ld_true)

mdic1 = {'data1': data1, 'data': data, 'norm_ld': norm_ld, 'ld_true': ld_true, 'u_true': u_true, 'R_true': R_true}
name1 = 'Data.mat'
scipy.savemat(name1, mdic1)




