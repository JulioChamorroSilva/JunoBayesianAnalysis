#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 07:26:50 2021

@author: P.Chimenti

    This code tests the base class of WoodHardness analysis

"""

import numpy as np

from Tests.BasicTools import WoodHardness_base as whb

samples = 10000

wh_base = whb.WoodHardness_base()

print("Number of samples: ",len(wh_base.Data_x))
mask = np.random.choice(a=[False, True], size=len(wh_base.Data_x), p=[0.5, 1-0.5])
print(mask)
wh_base.reset(mask)
print(wh_base.Data_x)
print(wh_base.Data_x_fit)
print(wh_base.Data_x_test)

samples, blobs = wh_base.run_mcmc(nsamples = samples)
np.save('test_wh_base.npy', np.concatenate((samples, blobs), axis=1) )
