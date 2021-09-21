#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:15:05 2021

@author: P.Chimenti

This code runs the mcmc of the Wood Hardness Bayesian analysis.
 
"""
import numpy as np
import pandas as pd 
import pyarrow as pa
import pyarrow.parquet as pq

from Tests.BasicTools import test_WoodHardness_linearCE as wh_lce

stat_factor = 10
thin = 10

wh_l1 = wh_lce.linear_ce()
nparams = 3

for i in range(len(wh_l1.Data)):
    print("Running ",i)
    mask = np.full(len(wh_l1.Data), True)
    mask[i]=False
    wh_l1.reset(itheta = 10.*np.ones(nparams), step_factor = 1., nsamples = 1000*stat_factor, nblobs = (2*len(wh_l1.Data)+2), data_mask = mask )
    samples, blobs = wh_l1.run_mcmc(thin_by = thin)
    df = pd.DataFrame(np.concatenate((samples, blobs), axis=1))
    table = pa.Table.from_pandas(df)
    pq.write_table(table, 'test_wh_mcmc_silly_'+'{num:02d}'.format(num=i)+'.parquet')


mask = np.full(len(wh_l1.Data), True)
wh_l1.reset(itheta = 10.*np.ones(nparams), step_factor = 1., nsamples = 10000*stat_factor, nblobs = (2*len(wh_l1.Data)+2), data_mask = mask )
samples, blobs = wh_l1.run_mcmc(thin_by = thin)
df = pd.DataFrame(np.concatenate((samples, blobs), axis=1))
table = pa.Table.from_pandas(df)
pq.write_table(table, 'test_wh_mcmc_silly_full.parquet')

