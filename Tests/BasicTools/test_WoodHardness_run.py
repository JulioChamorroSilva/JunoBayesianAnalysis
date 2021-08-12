#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:15:05 2021

@author: P.Chimenti

This code runs the mcmc of the Wood Hardness Bayesian analysis.
 
"""

import shelve

from Tests.BasicTools import test_WoodHardness_mcmc as WH_mcmc

wh_m = WH_mcmc.WoodHardness_mcmc()
wh_m.init(nparams = 2000, nsamples = 1000, nblobs = 10000)
samples, blobs = wh_m.run_mcmc()


d = shelve.open("test_wh_mcmc_silly.db")
d["samples"] = samples
d["blobs"]   = blobs
d.close()