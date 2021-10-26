#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 07:26:50 2021

@author: P.Chimenti

    This class implement the linear model with constant absolute error for the Wood Hardness test

"""

import argparse
import numpy as np
from   matplotlib import pyplot as plt

from Tests.BasicTools import WoodHardness_linearCE as whlce
from Tools import PlotChain as pc
from Tools import MonitoringChainVariable as MCV
from Tools import KDE_1d 
from Tools import KDE_2d 


parser = argparse.ArgumentParser(description='Tests the WoodHardness linear CE model.')
parser.add_argument('--run-mcmc', dest='run_mcmc', action='store_true', help="Enable running the MCMC")
parser.add_argument('--no-run-mcmc', dest='run_mcmc', action='store_false', help="Disable running the MCMC")
parser.add_argument('--show', dest='show', action='store_true', help="Enable plt.show")
parser.add_argument('--no-show', dest='show', action='store_false', help="Disable plt.show")
parser.add_argument("-n", "--stat-factor", type=int, dest='stat_factor',
                    help="Samples for each walker")
parser.add_argument("-f", "--file-tag", type=str, dest='file_tag',
                    help="Tag used in files")
parser.add_argument("-s", "--skip", type=int, dest='skip',
                    help="Number of samples to skip in analysis")

parser.set_defaults(run_mcmc     = True)
parser.set_defaults(show         = False)
parser.set_defaults(stat_factor  = 10000)
parser.set_defaults(file_tag     = "")
parser.set_defaults(skip         = 1000)

args = parser.parse_args()
print("Testing Wood Hardness linearCE with the following arguments:")
print(args)

walkers = 16

linearCE_runner = whlce.WoodHardness_linearCE()


if args.run_mcmc:
    #mask = np.random.choice(a=[False, True], size=len(linearCE_runner.Data_x), p=[0.5, 1-0.5])
    mask = [True]*len(linearCE_runner.Data_x)
    linearCE_runner.reset(mask)
    samples, blobs = linearCE_runner.run_mcmc(itheta = 500*np.ones(3), nsamples = args.stat_factor)
    np.save('test_wh_linearCE_'+args.file_tag+'.npy', np.concatenate((samples, blobs), axis=1) )

    for i in range(len(linearCE_runner.Data_x)):
        print("Removing element: ",i)
        mask_cross_val = [True]*len(linearCE_runner.Data_x)
        mask_cross_val[i]=False
        print(mask_cross_val)
        linearCE_runner.reset(mask)
        samples, blobs = linearCE_runner.run_mcmc(itheta = 500*np.ones(3), nsamples = args.stat_factor)
        np.save('test_wh_linearCE_'+args.file_tag+'_'+str(i).zfill(2)+'.npy', np.concatenate((samples, blobs), axis=1) )

    
data = np.load('test_wh_linearCE_'+args.file_tag+'.npy')
#print(data[0,:])
print("Plotting chains...")
apc = pc.PlotChain()
apc.Plot(data[:,0], "m", file_tag = args.file_tag)
apc.Plot(data[:args.skip,0], "m", file_tag = args.file_tag+"_zoom")
apc.Plot(data[:,1], "b", file_tag = args.file_tag)
apc.Plot(data[:args.skip,1], "b", file_tag = args.file_tag+"_zoom")
apc.Plot(data[:,2], "sigma", file_tag = args.file_tag)
apc.Plot(data[:args.skip,2], "sigma", file_tag = args.file_tag+"_zoom")
print("Done!")

print("Monitoring Variables...")
chains_means = []
chains_means_err = []
chains_stdevs = []
chains_stdevs_err = []
discard_factor = args.skip/len(data[:,0])
print("mean \t meanerr \t std \t stderr")

mon_m = MCV.Monitor(data[:,0], walkers, discard_factor)
chains_means.append(round(mon_m.mean,4))
chains_means_err.append(round(mon_m.mean_err,4))
chains_stdevs.append(round(mon_m.std,4))
chains_stdevs_err.append(round(mon_m.std_err,4))
print("m: ",chains_means[-1],chains_means_err[-1],chains_stdevs[-1],chains_stdevs_err[-1])
mon_b = MCV.Monitor(data[:,1], walkers, discard_factor)
chains_means.append(round(mon_b.mean,4))
chains_means_err.append(round(mon_b.mean_err,4))
chains_stdevs.append(round(mon_b.std,4))
chains_stdevs_err.append(round(mon_b.std_err,4))
print("b: ",chains_means[-1],chains_means_err[-1],chains_stdevs[-1],chains_stdevs_err[-1])
mon_s = MCV.Monitor(data[:,2], walkers, discard_factor)
chains_means.append(round(mon_s.mean,4))
chains_means_err.append(round(mon_s.mean_err,4))
chains_stdevs.append(round(mon_s.std,4))
chains_stdevs_err.append(round(mon_s.std_err,4))
print("s: ",chains_means[-1],chains_means_err[-1],chains_stdevs[-1],chains_stdevs_err[-1])


fig, axs = plt.subplots(3, 3, figsize=(20, 20))

fig.delaxes(axs[0,1])
fig.delaxes(axs[0,2])
fig.delaxes(axs[1,2])

kde_var0 = KDE_1d.KDE_1d(data[args.skip:,0])
kde_var0.Plot(axs[0,0])
kde_var1 = KDE_1d.KDE_1d(data[args.skip:,1])
kde_var1.Plot(axs[1,1])
kde_var2 = KDE_1d.KDE_1d(data[args.skip:,2])
kde_var2.Plot(axs[2,2])

kde_1v0 = KDE_2d.KDE_2d(data[args.skip:,0],data[args.skip:,1])
kde_1v0.Plot(axs[1,0])
kde_2v0 = KDE_2d.KDE_2d(data[args.skip:,0],data[args.skip:,2])
kde_2v0.Plot(axs[2,0])
kde_2v1 = KDE_2d.KDE_2d(data[args.skip:,1],data[args.skip:,2])
kde_2v1.Plot(axs[2,1])

fig.savefig("TestTriangle_WH_LinearCE"+args.file_tag+".png")
plt.clf()
print("Done!")



