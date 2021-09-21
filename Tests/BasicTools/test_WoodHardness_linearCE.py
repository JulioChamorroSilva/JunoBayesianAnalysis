#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 07:26:50 2021

@author: P.Chimenti

    This class implement the linear model with constant absolute error for the Wood Hardness test

"""

import numpy as np

from Tests.BasicTools import test_WoodHardness_mcmc as wh_mcmc

class linear_ce(wh_mcmc.WoodHardness_mcmc):
        def __init__(self):
            print("Hello Linear Model Constant Error!")


        def reset(self, data_mask, **kwargs):
            """ This function extendes the base to allow for fit and test data selection """            
            self.data_fit = self.Data[data_mask]
            self.data_fit_x = self.data_fit[:,0]
            self.data_fit_y = self.data_fit[:,1]
            self.data_test= self.Data[np.logical_not(data_mask)]
            self.data_test_x = self.data_fit[:,0]
            self.data_test_y = self.data_fit[:,1]
            super(linear_ce,self).reset(**kwargs)

        MAX_M = 1000    # used as prior bounds
        MAX_B = 10000   
        MAX_S = 1000


        def log_likelihood(self, theta):
            m, b, sigma   = theta      # we need 3 parameters! The error is unknown
            model         = m * self.data_fit_x + b
            return -0.5 * np.sum((self.data_fit_y - model) ** 2 / (sigma**2) ) - len(self.data_fit_x)*np.log(sigma) - len(self.data_fit_x) * np.log(np.sqrt(2 * np.pi))



        def log_prior(self, theta):
            m, b, sigma = theta
            if -1.*self.MAX_M < m < self.MAX_M and -1.*self.MAX_B < b < self.MAX_B and 0. < sigma < self.MAX_S:
                return 0.0
            return -np.inf


        def log_probability(self, theta):
            """ Posterior probability
            for now just gaussian centered in 1 and sigma 1 """        
            lp = self.log_prior(theta)
            if not np.isfinite(lp):
                result = [-np.inf] 
                result.extend(np.zeros(self.nblobs))
                return result

            # now calculate random samples and pointwise log predictive density (fit and test data!)
            ppc = np.zeros(len(self.Data_x))
            model = theta[0] * self.Data_x + theta[1]
            for i in range(len(self.Data_x)):
                ppc[i] = model[i] + np.random.normal(0., theta[2])
            plpd = (-0.5 * (self.Data_y - model) ** 2 / (theta[2]**2) ) - np.log(theta[2]) - np.log(np.sqrt(2 * np.pi)) 
            
            # now log-likelihood
            ll = self.log_likelihood(theta)
            
            # pack result and return
            result = [lp+ll, lp, ll]
            result.extend(ppc)
            result.extend(plpd)
            return result


            


