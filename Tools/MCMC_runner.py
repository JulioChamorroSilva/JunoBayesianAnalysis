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
    
    def __init__(self, nparams = 3, nwalkers = 16, move_cov = []):
        """ This function initialize the emcee ensemble sampler: 80% stretch and 20% gaussian
        the move_cov is an "estimate" of the posterior parameters covariance diagonal elements. 
        (for exmple from likelihood studies)""" 
        self.nparams  = nparams
        self.nwalkers = nwalkers
        if not any(move_cov) :
            move_cov   = np.ones(self.nparams)
        self.move_cov = move_cov
        self.sampler = emcee.EnsembleSampler(
                nwalkers,
                nparams,
                self.log_probability,
                    moves=[
                            (emcee.moves.StretchMove(live_dangerously = True), 0.8),
                            (emcee.moves.GaussianMove((self.move_cov/nparams),  mode="vector"), 0.2)
                            ],
                )
    
    def log_probability(self, theta):
        """ Distribution to be samples. To be reimplemented by inheritance.
        it includes 3 blobs all equal 1 just for testing purposes. """        
        nblobs = 3
        log_prob = -1.0*(sum(theta**2))/2. 
        result = [log_prob] 
        result.extend(np.ones(nblobs))
        return result
                
    def run_mcmc(self, itheta = [], nsamples = 100):
        """ Here we actually run the MCMC.
        itheta is the initial seed for samplers.
        Samplers are initialized adding to this seed a random disperion according to move_cov.
        The total number of samples is nsample times nwalkers.
        This function returns samples of parameters and blobs.        
        """
        if not any(itheta):
            itheta = np.zeros(self.nparams)
        thin_by = self.nparams
        iwalkers = np.array(itheta)+self.move_cov*np.random.randn(self.nwalkers,self.nparams)
        self.sampler.run_mcmc(iwalkers, nsamples, progress=True, skip_initial_state_check = True, thin_by = thin_by)
        flat_samples = self.sampler.get_chain(flat=True)
        blobs        = self.sampler.get_blobs(flat=True)
        return flat_samples, blobs

    


