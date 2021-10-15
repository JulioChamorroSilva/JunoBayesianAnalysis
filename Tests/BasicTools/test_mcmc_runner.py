#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 07:23:54 2021

@author: P.Chimenti
"""

import argparse
import numpy as np
from   matplotlib import pyplot as plt

from Tools import MCMC_runner as mr
from Tools import PlotChain as pc
from Tools import MonitoringChainVariable as MCV
from Tools import KDE_1d 
from Tools import KDE_2d 


parser = argparse.ArgumentParser(description='Tests the MCMC_runner class from Tools module.')
parser.add_argument('--run-mcmc', dest='run_mcmc', action='store_true', help="Enable running the MCMC")
parser.add_argument('--no-run-mcmc', dest='run_mcmc', action='store_false', help="Disable running the MCMC")
parser.add_argument('--show', dest='show', action='store_true', help="Enable plt.show")
parser.add_argument('--no-show', dest='show', action='store_false', help="Disable plt.show")
parser.add_argument("-p", "--params", type=int, dest='nparams',
                    help="Number of parameters to be considered")
parser.add_argument("-n", "--stat-factor", type=int, dest='stat_factor',
                    help="Samples for each walker")
parser.add_argument("-w", "--walkers", type=int, dest='walkers',
                    help="Number of walkers used in MCMC")
parser.add_argument("-f", "--file-tag", type=str, dest='file_tag',
                    help="Tag used in files")
parser.add_argument("-s", "--skip", type=int, dest='skip',
                    help="Number of samples to skip in analysis")

parser.set_defaults(run_mcmc     = True)
parser.set_defaults(show         = False)
parser.set_defaults(nparams      = 3)
parser.set_defaults(stat_factor  = 1000)
parser.set_defaults(walkers      = 16)
parser.set_defaults(file_tag     = "")
parser.set_defaults(skip         = 16000)

args = parser.parse_args()
print("Testing MCMC_runner with the following arguments:")
print(args)

runner = mr.MCMC_runner(nparams = args.nparams, nwalkers = args.walkers)

# first run
if args.run_mcmc:
    samples, blobs = runner.run_mcmc(itheta = 10.*np.ones(args.nparams), nsamples = args.stat_factor)
    np.save('test_runner_mcmc_'+args.file_tag+'.npy', np.concatenate((samples, blobs), axis=1) )

data = np.load('test_runner_mcmc_'+args.file_tag+'.npy')

print("Plotting chains...")
apc = pc.PlotChain()
apc.Plot(data[:,0], "var_0", file_tag = args.file_tag)
apc.Plot(data[:args.skip,0], "var_0", file_tag = args.file_tag+"_zoom")
apc.Plot(data[:,1], "var_1", file_tag = args.file_tag)
apc.Plot(data[:args.skip,1], "var_1", file_tag = args.file_tag+"_zoom")
apc.Plot(data[:,2], "var_2", file_tag = args.file_tag)
apc.Plot(data[:args.skip,2], "var_2", file_tag = args.file_tag+"_zoom")
apc.Plot(data[:,3], "var_3", file_tag = args.file_tag)
apc.Plot(data[:args.skip,3], "var_3", file_tag = args.file_tag+"_zoom")
print("Done!")


print("Monitoring Variables...")
chains_means = []
chains_means_err = []
chains_stdevs = []
chains_stdevs_err = []
discard_factor = args.skip/len(data[:,0])
print("mean \t meanerr \t std \t stderr")
for i in range(args.nparams):
    mon = MCV.Monitor(data[:,i], args.walkers, discard_factor)
    chains_means.append(round(mon.mean,4))
    chains_means_err.append(round(mon.mean_err,4))
    chains_stdevs.append(round(mon.std,4))
    chains_stdevs_err.append(round(mon.std_err,4))
    print(chains_means[-1],chains_means_err[-1],chains_stdevs[-1],chains_stdevs_err[-1])

plt.hist(chains_means)
plt.savefig("ChainMeans_"+args.file_tag+".png")
if args.show : plt.show()
plt.clf()
plt.hist(chains_means_err)
plt.savefig("ChainMeansErr_"+args.file_tag+".png")
if args.show : plt.show()
plt.clf()
plt.hist(chains_stdevs)
plt.savefig("ChainStdevs_"+args.file_tag+".png")
if args.show : plt.show()
plt.clf()
plt.hist(chains_stdevs_err)
plt.savefig("ChainStdevErr_"+args.file_tag+".png")
if args.show : plt.show()
plt.clf()
print("Done!")
print("Triangle Plot...")



fig, axs = plt.subplots(4, 4, figsize=(20, 20))

fig.delaxes(axs[0,1])
fig.delaxes(axs[0,2])
fig.delaxes(axs[0,3])
fig.delaxes(axs[1,2])
fig.delaxes(axs[1,3])
fig.delaxes(axs[2,3])

kde_var0 = KDE_1d.KDE_1d(data[args.skip:,0])
kde_var0.Plot(axs[0,0])
kde_var1 = KDE_1d.KDE_1d(data[args.skip:,1])
kde_var1.Plot(axs[1,1])
kde_var2 = KDE_1d.KDE_1d(data[args.skip:,2])
kde_var2.Plot(axs[2,2])
kde_var3 = KDE_1d.KDE_1d(data[args.skip:,3])
kde_var3.Plot(axs[3,3])

kde_1v0 = KDE_2d.KDE_2d(data[args.skip:,0],data[args.skip:,1])
kde_1v0.Plot(axs[1,0])
kde_2v0 = KDE_2d.KDE_2d(data[args.skip:,0],data[args.skip:,1])
kde_2v0.Plot(axs[2,0])
kde_3v0 = KDE_2d.KDE_2d(data[args.skip:,0],data[args.skip:,1])
kde_3v0.Plot(axs[3,0])
kde_2v1 = KDE_2d.KDE_2d(data[args.skip:,0],data[args.skip:,1])
kde_2v1.Plot(axs[2,1])
kde_3v1 = KDE_2d.KDE_2d(data[args.skip:,0],data[args.skip:,1])
kde_3v1.Plot(axs[3,1])
kde_3v2 = KDE_2d.KDE_2d(data[args.skip:,0],data[args.skip:,1])
kde_3v2.Plot(axs[3,2])

fig.savefig("TestTriangle_"+args.file_tag+".png")
if args.show : plt.show()
plt.clf()
print("Done!")
