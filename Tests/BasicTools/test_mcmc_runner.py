#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 07:23:54 2021

@author: P.Chimenti
"""

import argparse
import numpy as np
from   matplotlib import pyplot as plt
from timeit import default_timer as timer

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

#print("Plotting chains...")
#apc = pc.PlotChain()
#apc.Plot(data[:,0], "var_0", file_tag = args.file_tag)
#apc.Plot(data[:args.skip,0], "var_0", file_tag = args.file_tag+"_zoom")
#apc.Plot(data[:,1], "var_1", file_tag = args.file_tag)
#apc.Plot(data[:args.skip,1], "var_1", file_tag = args.file_tag+"_zoom")
#apc.Plot(data[:,2], "var_2", file_tag = args.file_tag)
#apc.Plot(data[:args.skip,2], "var_2", file_tag = args.file_tag+"_zoom")
#print("Done!")
#
#
#print("Monitoring Variables...")
#chains_means = []
#chains_means_err = []
#chains_stdevs = []
#chains_stdevs_err = []
#
#discard_factor = args.skip/len(data[:,0])
#print("mean \t meanerr \t std \t stderr")
#for i in range(args.nparams):
#    mon = MCV.Monitor(data[:,i], args.walkers, discard_factor)
#    chains_means.append(round(mon.mean,4))
#    chains_means_err.append(round(mon.mean_err,4))
#    chains_stdevs.append(round(mon.std,4))
#    chains_stdevs_err.append(round(mon.std_err,4))
#    print(chains_means[-1],chains_means_err[-1],chains_stdevs[-1],chains_stdevs_err[-1])
#
#plt.hist(chains_means)
#if args.show : plt.show()
#plt.clf()
#plt.hist(chains_means_err)
#if args.show : plt.show()
#plt.clf()
#plt.hist(chains_stdevs)
#if args.show : plt.show()
#plt.clf()
#plt.hist(chains_stdevs_err)
#if args.show : plt.show()
#plt.clf()

fig, axs = plt.subplots(4, 4, figsize=(20, 20))

fig.delaxes(axs[0,1])
fig.delaxes(axs[0,2])
fig.delaxes(axs[0,3])
fig.delaxes(axs[1,2])
fig.delaxes(axs[1,3])
fig.delaxes(axs[2,3])

def PlotMarginalHist(ax, data):
    time00 = timer() 
    kde_1 = KDE_1d.KDE_1d(data)
    time01 = timer() 
    x,y,w = kde_1.getKDE()
    time02 = timer() 
    b     = kde_1.getBins()
    time03 = timer() 
    M_68, prob_68d     = kde_1.getPLMask(level=0.6827)
    time04 = timer() 
    M_95, prob_95d     = kde_1.getPLMask(level=0.9545)
    time05 = timer() 
    M_99, prob_99d     = kde_1.getPLMask(level=0.9973)
    time06 = timer() 
    y_68           = np.array([0 if not M_68[i] else y[i] for i in range(len(y)) ])
    time07 = timer() 
    y_95           = np.array([0 if not M_95[i] else y[i] for i in range(len(y)) ])
    time08 = timer() 
    y_99           = np.array([0 if not M_99[i] else y[i] for i in range(len(y)) ])
    time09 = timer() 
    ax.plot(x,y)
    time10 = timer() 
    print(len(b))
    print(len(y_99))
    ax.hist(b[:-1],b,weights=y_99,color='yellow')
    time11= timer() 
    ax.hist(b[:-1],b,weights=y_95,color='orange')
    time12 = timer() 
    ax.hist(b[:-1],b,weights=y_68,color='red')
    time13 = timer() 
    print("Timing kde 1d:")
    print("01 ",time01-time00)
    print("02 ",time02-time01)
    print("03 ",time03-time02)
    print("04 ",time04-time03)
    print("05 ",time05-time04)
    print("06 ",time06-time05)
    print("07 ",time07-time06)
    print("08 ",time08-time07)
    print("09 ",time09-time08)
    print("10 ",time10-time09)
    print("11 ",time11-time10)
    print("12 ",time12-time11)
    print("13 ",time13-time12)


def PlotJoint(ax, data1, data2):
    kde_2 = KDE_2d.KDE_2d(data1,data2)
    x,y,z = kde_2.getKDE(gridfactor = 1)
    M_2d, prob_2d_1 = kde_2.getPLMask(level=0.6827)
    M_2d, prob_2d_2 = kde_2.getPLMask(level=0.9545)
    M_2d, prob_2d_3 = kde_2.getPLMask(level=0.9973)
    mean_x=np.mean(data[:args.skip,0])
    mean_y=np.mean(data[:args.skip,1])
    ax.contourf(x,y,z,[0.,prob_2d_3,prob_2d_2,prob_2d_1,1.001*np.amax(z)],colors=('white','yellow','orange','red'))
    #plt.colorbar(cf,ax=ax)
    ax.plot(mean_x,mean_y,'ro') 

#PlotJoint(axs[1,0],data[args.skip:,0],data[args.skip:,1])
#PlotJoint(axs[2,0],data[args.skip:,0],data[args.skip:,2])
#PlotJoint(axs[3,0],data[args.skip:,0],data[args.skip:,3])
#PlotJoint(axs[2,1],data[args.skip:,1],data[args.skip:,2])
#PlotJoint(axs[3,1],data[args.skip:,1],data[args.skip:,3])
#PlotJoint(axs[3,2],data[args.skip:,2],data[args.skip:,3])

time00 = timer()
PlotMarginalHist(axs[0,0],data[args.skip:,0])
time01 = timer()
PlotMarginalHist(axs[1,1],data[args.skip:,1])
time02 = timer()
PlotMarginalHist(axs[2,2],data[args.skip:,2])
time03 = timer()
PlotMarginalHist(axs[3,3],data[args.skip:,3])
time04 = timer()
print("Timing total kde 1d:")
print("01 ",time01-time00)
print("02 ",time02-time01)
print("03 ",time03-time02)
print("04 ",time04-time03)


fig.savefig("TestTriangle.png")
#fig.tight_layout()
#plt.show(fig)
