#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 07:26:50 2021

@author: P.Chimenti

    This class implement the linear model with constant absolute error for the Wood Hardness test

"""

import numpy as np

from Tests.BasicTools import WoodHardness_linearCE as whlce

linearCE_runner = whlce.WoodHardness_linearCE()

samples = 10000
mask = np.random.choice(a=[False, True], size=len(linearCE_runner.Data_x), p=[0.5, 1-0.5])
print(mask)
linearCE_runner.reset(mask)
samples, blobs = linearCE_runner.run_mcmc(nsamples = samples)
np.save('test_wh_linearCE.npy', np.concatenate((samples, blobs), axis=1) )






