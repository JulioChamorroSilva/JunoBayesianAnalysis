#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 07:22:21 2021

@author: chimenti

This class provides a tool for a 2d KDE plot.
It needs two arrays of 1d measurements.

"""

import numpy as np
import scipy.optimize
from scipy.stats import norm 


class KDE_2d:
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.setInterval()
        self.points_x = np.array([])
        self.points_y = np.array([])
        self.bins_x   = np.array([])
        self.bins_y   = np.array([])
        self.vals   = np.array([]) 
        self.step_x   = 0
        self.step_y   = 0

    def setInterval(self):
        min_x = np.amin(self.data_x)
        max_x = np.amax(self.data_x)
        self.std_x = np.std(self.data_x)
        self.min_x = min_x-self.std_x
        self.max_x = max_x+self.std_x
        min_y = np.amin(self.data_y)
        max_y = np.amax(self.data_y)
        self.std_y = np.std(self.data_y)
        self.min_y = min_y-self.std_y
        self.max_y = max_y+self.std_y

    
    def getKDE(self, bandwidth_x = 0, bandwidth_y = 0, gridfactor = 5):
        if bandwidth_x == 0:
            bandwidth_x = 1.06*self.std_x/(len(self.data_x)**(1.0/5))
        if bandwidth_y == 0:
            bandwidth_y = 1.06*self.std_y/(len(self.data_y)**(1.0/5))

        self.step_x   = bandwidth_x/gridfactor
        self.step_y   = bandwidth_y/gridfactor
        xx = np.arange(self.min_x,self.max_x+self.step_x,self.step_x)
        yy = np.arange(self.min_y,self.max_y+self.step_y,self.step_y)
        self.points_x, self.points_y = np.meshgrid(xx, yy)
        xx = xx - self.step_x/2
        xx = np.append(xx, [xx[-1]+self.step_x])
        yy = yy - self.step_y/2
        yy = np.append(yy, [yy[-1]+self.step_y])
        self.bins_x, self.bins_y = np.meshgrid(xx, yy)
        self.vals = np.zeros(xx.shape)

        spread_factors = [norm.cdf(-5 + ((i+0.5)/gridfactor))-norm.cdf(-5 + ((i-0.5)/gridfactor)) for i in range(10*gridfactor+1)]

        self.vals = np.histogram2d(self.data_x, self.data_y, bins = (xx,yy))[0]
        self.vals = self.vals.T

        spreaded = np.zeros(self.vals.shape)
        shape_x, shape_y = spreaded.shape 
        for i in range(10*gridfactor+1):
            for j in range(10*gridfactor+1):
                spreaded[i:shape_x-10*gridfactor+i,j:shape_y-10*gridfactor+j] += \
                    spread_factors[i]*spread_factors[j]*self.vals[5*gridfactor:-(5*gridfactor),5*gridfactor:-(5*gridfactor)] 
        self.vals = spreaded

        normalization = np.sum(self.vals)*self.step_x*self.step_y
        self.vals = self.vals/normalization

        return self.points_x, self.points_y, self.vals, self.step_x, self.step_y


    def objective( self, density, prob):
        mask = self.vals>density
        count = self.vals[mask]*self.step_x*self.step_y
        return count.sum() - prob


    def getPLMask( self, level = 0.9):
        prob = scipy.optimize.bisect(self.objective, self.vals.min(), self.vals.max(), args=(level,))
        mask = self.vals>prob
        return mask, prob
        
        