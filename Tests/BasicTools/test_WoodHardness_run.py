#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 13:15:05 2021

@author: P.Chimenti
"""

import shelve

from Tests.BasicTools import test_WoodHardness_mcmc as WH_mcmc

wh_m = WH_mcmc.WoodHardness_mcmc()
wh_m.init()
samples, blobs = wh_m.run_mcmc()


d = shelve.open("test_wh_mcmc_silly.db")
d["samples"] = samples
d["blobs"]   = blobs
d.close()
