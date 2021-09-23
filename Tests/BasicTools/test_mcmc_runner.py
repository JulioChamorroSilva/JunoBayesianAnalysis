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


parser = argparse.ArgumentParser(description='Tests the MCMC_runner class from Tools module.')
parser.add_argument('--run-mcmc', dest='run_mcmc', action='store_true', help="Enable running the MCMC")
parser.add_argument('--no-run-mcmc', dest='run_mcmc', action='store_false', help="Disable running the MCMC")
parser.add_argument("-p", "--params", type=int, dest='nparams',
                    help="Number of parameters to be considered")
parser.add_argument("-n", "--stat-factor", type=int, dest='stat_factor',
                    help="Samples for each walker")
parser.add_argument("-b", "--blobs", type=int, dest='blobs',
                    help="blobs to be considered (not 1)")
parser.add_argument("-t", "--thin", type=int, dest='thin',
                    help="increase number of samples by this factor, but store only stats-factor per walker")
parser.add_argument("-w", "--walkers", type=int, dest='walkers',
                    help="Number of walkers used in MCMC")
parser.add_argument("-f", "--file-tag", type=str, dest='file_tag',
                    help="Tag used in files")
parser.add_argument("-s", "--skip", type=int, dest='skip',
                    help="Number of samples to skip in analysis")
parser.add_argument("-m", "--move-factor", type=float, dest='move_factor',
                    help="Numeric factor of gaussian move")

parser.set_defaults(run_mcmc     = True)
parser.set_defaults(nparams      = 3)
parser.set_defaults(stat_factor  = 1000)
parser.set_defaults(blobs        = 2)
parser.set_defaults(thin         = 1)
parser.set_defaults(walkers      = 16)
parser.set_defaults(file_tag     = "")
parser.set_defaults(skip         = 16000)
parser.set_defaults(move_factor  = 1.)


args = parser.parse_args()
print("Testing MCMC_runner with the following arguments:")
print(args)


runner = mr.MCMC_runner()

# first run
if args.run_mcmc:
    runner.reset(itheta = 10.*np.ones(args.nparams), step_factor = args.move_factor, nsamples = args.stat_factor, nblobs = args.blobs, nwalkers = args.walkers)
    samples, blobs = runner.run_mcmc(thin_by = args.thin)
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
print("Done!")


print("Monitoring Variables...")
discard_factor = args.skip/len(data[:,0])
mon0 = MCV.Monitor(data[:,0], args.walkers, discard_factor)
mon1 = MCV.Monitor(data[:,1], args.walkers, discard_factor)
mon2 = MCV.Monitor(data[:,2], args.walkers, discard_factor)
print("means:")
print(round(mon0.mean,3)," PM ", round(mon0.mean_err,3)) 
print(round(mon1.mean,3)," PM ", round(mon1.mean_err,3)) 
print(round(mon2.mean,3)," PM ", round(mon2.mean_err,3)) 
print("vars:")
print(round(mon0.std,3)," PM ", round(mon0.std_err,3)) 
print(round(mon1.std,3)," PM ", round(mon1.std_err,3)) 
print(round(mon2.std,3)," PM ", round(mon2.std_err,3)) 
print("reduction Factors:")
print(round(mon0.R,5))
print(round(mon1.R,5))
print(round(mon2.R,5))