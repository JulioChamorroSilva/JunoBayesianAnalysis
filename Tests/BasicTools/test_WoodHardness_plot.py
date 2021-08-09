#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:32:49 2021

@author: chimenti
"""

import shelve

d = shelve.open("test_wh_mcmc_silly.db")
print(list(d))
samples = d["samples"]
blobs   = d["blobs"]
print(samples)
print(blobs)
