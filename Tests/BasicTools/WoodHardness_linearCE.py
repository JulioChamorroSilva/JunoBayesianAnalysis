# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 08:48:28 2021

@author: chimenti

    This class implement the linear model with constant absolute error for the Wood Hardness analysis

"""


import numpy as np

from Tests.BasicTools import WoodHardness_base as whb

class WoodHardness_linearCE( whb.WoodHardness_base ):
    """ This class implements the linear model with constant absolute error for the Wood Hardness dataset """
    
    MAX_M = 1000    # used as prior bounds
    MAX_B = 10000   
    MAX_S = 1000

    
    def __init__(self):
        move_cov=np.array([100,1000,100])
        whb.WoodHardness_base.__init__(self, nparams=3, nwalkers=16, move_cov=move_cov)
        
    def reset(self, data_mask=[]):
        whb.WoodHardness_base.reset(self, data_mask=data_mask)

    def log_likelihood(self, theta):
        m, b, sigma   = theta      # we need 3 parameters! The error is unknown
        model_fit         = m * self.Data_x_fit + b
        model_test        = m * self.Data_x_test + b
        self.fit_loglikes  = -0.5*( (self.Data_y_fit  - model_fit ) ** 2 / (sigma**2) ) - np.log(sigma) - np.log(np.sqrt(2 * np.pi))
        self.test_loglikes = -0.5*( (self.Data_y_test - model_test) ** 2 / (sigma**2) ) - np.log(sigma) - np.log(np.sqrt(2 * np.pi))
        self.fit_randomsample   = model_fit  + np.random.normal(scale = sigma, size = len(model_fit ))
        self.test_randomsample  = model_test + np.random.normal(scale = sigma, size = len(model_test))
        return np.sum(self.fit_loglikes)

    def log_prior(self, theta):
        m, b, sigma = theta
        if -1.*self.MAX_M < m < self.MAX_M and -1.*self.MAX_B < b < self.MAX_B and 0. < sigma < self.MAX_S:
            return 0.0
        return -np.inf

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            result = [-np.inf] 
            result.extend(np.zeros(self.nblobs))
            return result
        ll = self.log_likelihood(theta)
        # pack result and return
        result = [lp+ll, lp, ll]
        result.extend(self.fit_loglikes)
        result.extend(self.fit_randomsample)
        result.extend(self.test_loglikes)
        result.extend(self.test_randomsample)
        return np.array(result)
    