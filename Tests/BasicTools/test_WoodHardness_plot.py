#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:32:49 2021

@author: P.Chimenti

This code make plots and summary tables of the bayesian Wood Hardness analysis.

"""

import shelve

d = shelve.open("test_wh_mcmc_silly.db")
print(list(d))
samples = d["samples"]
blobs   = d["blobs"]
print(samples.shape)
print(blobs.shape)
