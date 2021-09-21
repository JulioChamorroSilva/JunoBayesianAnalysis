#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:32:49 2021

@author: P.Chimenti

This code make plots and summary tables of the bayesian Wood Hardness analysis.

"""

import matplotlib.pyplot as plt 
import numpy as np
import pyarrow.parquet as pq

from Tools import KDE_1d 
from Tools import KDE_2d 
from Tools import MonitoringChainVariable as MCV

def PlotChain( data, name = "var1"):
    fig, ax = plt.subplots()
    ax.plot( np.arange(start=1, stop=len(data)+1), data)
    ax.set(xlabel='iteration', ylabel=name,
       title='Chain - '+name)
    ax.grid()

    fig.savefig("Chain_"+name+".png")
    plt.close()

table = pq.read_table('test_wh_mcmc_silly_full.parquet')
df    =  table.to_pandas()
arr   =  df.to_numpy()
print(df)

PlotChain(arr[:50000,0], name = "Var0")
PlotChain(arr[:50000,1], name = "Var1")
PlotChain(arr[:50000,2], name = "Var2")

#print("mon0")
#mon0 = MCV.Monitor(arr[:,0], 16, 0.5)
#print("mon1")
#mon1 = MCV.Monitor(arr[:,1], 16, 0.5)
#print("mon2")
#mon2 = MCV.Monitor(arr[:,2], 16, 0.5)
#print("plotting...")
#fig, (ax1,ax2,ax3) = plt.subplots(3)
#print("plot 1")
#ax1.plot( np.arange(len(mon0.data)), mon0.data)
#ax1.plot( np.arange(len(mon1.data)), mon1.data)
#ax1.plot( np.arange(len(mon2.data)), mon2.data)
#print("plot 2")
#ax2.plot( np.arange(len(mon0.V_t)) , mon0.V_t )
#ax2.plot( np.arange(len(mon1.V_t)) , mon1.V_t )
#ax2.plot( np.arange(len(mon2.V_t)) , mon2.V_t )
#print("plot 3")
#ax3.plot( np.arange(len(mon0.n_eff)) , mon0.n_eff )
#ax3.plot( np.arange(len(mon1.n_eff)) , mon1.n_eff )
#ax3.plot( np.arange(len(mon2.n_eff)) , mon2.n_eff )
#print("plot show")
#plt.show()
#print("reduction Factors:")
#print(mon0.R)
#print(mon1.R)
#print(mon2.R)
#print("n_eff:")
#print(mon0.n_eff[len(mon0.n_eff)//2])
#print(mon1.n_eff[len(mon1.n_eff)//2])
#print(mon2.n_eff[len(mon2.n_eff)//2])
#
#full_sample = len(arr[:,0])
#
#kde_1 = KDE_1d.KDE_1d(arr[full_sample//2:-1,0])
#x,y,w = kde_1.getKDE()
#b     = kde_1.getBins()
#M_68, prob_68d     = kde_1.getPLMask(level=0.6827)
#M_95, prob_95d     = kde_1.getPLMask(level=0.9545)
#M_99, prob_99d     = kde_1.getPLMask(level=0.9973)
#y_68           = np.array([0 if not M_68[i] else y[i] for i in range(len(y)) ])
#y_95           = np.array([0 if not M_95[i] else y[i] for i in range(len(y)) ])
#y_99           = np.array([0 if not M_99[i] else y[i] for i in range(len(y)) ])
#plt.plot(x,y)
#plt.hist(b[:-1],b,weights=y_99,color='yellow')
#plt.hist(b[:-1],b,weights=y_95,color='orange')
#plt.hist(b[:-1],b,weights=y_68,color='red')
#plt.show()

#kde_2 = KDE_2d.KDE_2d(arr[full_sample//2:-1,0],arr[full_sample//2:-1,1])
#x,y,z = kde_2.getKDE(gridfactor = 1)
#M_2d, prob_2d_1 = kde_2.getPLMask(level=0.6827)
#M_2d, prob_2d_2 = kde_2.getPLMask(level=0.9545)
#M_2d, prob_2d_3 = kde_2.getPLMask(level=0.9973)
#mean_x=np.mean(arr[full_sample//2:-1,0])
#mean_y=np.mean(arr[full_sample//2:-1,1])
#print(prob_2d_1)
#print(prob_2d_2)
#print(prob_2d_3)
#print(np.amax(z))
#cf = plt.contourf(x,y,z,[0.,prob_2d_3,prob_2d_2,prob_2d_1,1.001*np.amax(z)],colors=('white','yellow','orange','red'))
#plt.colorbar(cf)
#plt.plot(mean_x,mean_y,'ro') 
#plt.show()



