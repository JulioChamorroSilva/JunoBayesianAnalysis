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
print(data[0,:])
print("Plotting chains...")
apc = pc.PlotChain()
apc.Plot(data[:,0], "m", file_tag = args.file_tag)
apc.Plot(data[:args.skip,0], "m", file_tag = args.file_tag+"_zoom")
apc.Plot(data[:,1], "b", file_tag = args.file_tag)
apc.Plot(data[:args.skip,1], "b", file_tag = args.file_tag+"_zoom")
apc.Plot(data[:,2], "sigma", file_tag = args.file_tag)
apc.Plot(data[:args.skip,2], "sigma", file_tag = args.file_tag+"_zoom")
print("Done!")






