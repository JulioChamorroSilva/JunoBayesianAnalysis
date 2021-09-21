#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 08:31:33 2021

@author: P.Chimenti

This code models a basic class for running bayesian MCMC. It acts as interface to emcee 

"""

import numpy as np
import emcee

class MCMC_runner:
    """ This is a base clas for running MCMC for bayesian analysis using emcee """
    
    def __init__(self):
        print("Hello WoodHardness_mcmc!")
    
    def log_probability(self, theta):
        """ Distribution to be samples. To be reimplemented by inheritance. """        
        log_prob = -1.0*(sum(theta**2))/2. 
        result = [log_prob] 
        if self.nblobs != 0 : result.extend(np.ones(self.nblobs))
        return result
        
    def reset(self, nparams = 3, itheta = [], nwalkers = 16, nsamples = 100, nblobs = 1, step_factor = 1.0, move_cov = []):
        """ This function set initial configuration to run the mcmc 
        nparams : number of parameters
        itheta : parameters initial values
        nwalkers : number of walker for ensamble sampling
        nsamples : sample of mcmc
        nblobs : number of blobs (see emcee docs)
        step_factor :  to adjust the step size
        move_cov : covariance of the gaussian move """
        if not any(itheta) :
            self.nparams = nparams
            self.itheta  = np.zeros(nparams)
        else :
            self.nparams = len(itheta)
            self.itheta  = itheta
        self.nwalkers = nwalkers
        if not any(move_cov) :
            self.move_cov   = step_factor*np.ones(self.nparams)
        self.step_factor    = step_factor
        self.nsamples       = nsamples
        self.nblobs         = nblobs
        
    def run_mcmc(self, thin_by = 1):
        """ Here we actually run the MCMC with a mixture of strech (80%) and gaussian (20%) moves.
        The function returns parameters samples and blobs 
        
        Thin_by : multiply this number to the number of samples to save to get the total samples
        
        """
        iwalkers = np.array(self.itheta)+self.move_cov*np.random.randn(self.nwalkers,self.nparams)
        nwalkers, ndim     = iwalkers.shape
        sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                self.log_probability,
                    moves=[
                            (emcee.moves.StretchMove(live_dangerously = True), 0.8),
                            (emcee.moves.GaussianMove(self.move_cov,  mode="vector"), 0.2)
                            ],
                )
        sampler.run_mcmc(iwalkers, self.nsamples, progress=True, skip_initial_state_check = True, thin_by = thin_by)
        flat_samples = sampler.get_chain(flat=True)
        blobs        = sampler.get_blobs(flat=True)
        return flat_samples, blobs

    


