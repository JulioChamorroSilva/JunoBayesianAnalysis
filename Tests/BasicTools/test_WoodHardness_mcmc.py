#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 08:31:33 2021

@author: P.Chimenti

This code models a bayesian mcmc on the data about wood hardness from table 15.5 of Dekking, et.al, "A Modern Introduction to
    Probability and Statistics" 

"""

import numpy as np
import matplotlib.pyplot as plt
import emcee
from scipy.optimize import minimize
import scipy.stats as st
import arviz as az
import argparse

class WoodHardness_mcmc:
    """ This class allow to run bayesian MCMC to analisez thw wood hardness data """
    
    """ The wood hardness data: """

    Data = np.array([
            [24.7,484],
            [24.8,427],
            [27.3,413],
            [28.4,517],
            [28.4,549],
            [29.0,648],
            [30.3,587],
            [32.7,704],
            [35.6,979],
            [38.5,914],
            [38.8,1070],
            [39.3,1020],
            [39.4,1210],
            [39.9,989],
            [40.3,1160],
            [40.6,1010],
            [40.7,1100],
            [40.7,1130],
            [42.9,1270],
            [45.8,1180],
            [46.9,1400],
            [48.2,1760],
            [51.5,1710],
            [51.5,2010],
            [53.4,1880],
            [56.0,1980],
            [56.5,1820],
            [57.3,2020],
            [57.6,1980],
            [59.2,2310],
            [59.8,1940],
            [66.0,3260],
            [67.4,2700],
            [68.8,2890],
            [69.1,2740],
            [69.1,3140]])
    Data_x = Data[:,0]
    Data_y = Data[:,1]

    def __init__(self):
        print("Hello WoodHardness_mcmc!")

    def log_likelihood(self, theta):
        return 0
    
    def log_prior(self, theta):
        return 0
    
    def log_probability(self, theta):
        prob = 0
        result = [prob] 
        result.extend(np.ones(self.nblobs))
        return result
        
    def init(self, nparams = 3, nsamples = 100, nblobs = 1, spread_factor = 1e-4):
        self.theta         = np.zeros(nparams)
        self.move_cov      = spread_factor*np.ones(nparams)
        self.spread_factor = spread_factor
        self.nsamples      = nsamples
        self.nblobs        = nblobs
        
    def run_mcmc(self):
        pos = self.theta + self.spread_factor * np.random.randn(1, len(self.theta))
        nwalkers, ndim = pos.shape
        #        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability)
        sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                self.log_probability,
                moves = emcee.moves.GaussianMove(self.move_cov)
                )
        sampler.run_mcmc(pos, self.nsamples, progress=True, skip_initial_state_check = True);
        flat_samples = sampler.get_chain(flat=True)
        blobs        = sampler.get_blobs(flat=True)
        return flat_samples, blobs

    


