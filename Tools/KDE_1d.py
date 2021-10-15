#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 07:22:21 2021

@author: chimenti

This class provides a tool for a 1d KDE plot.
It needs an array of 1d measurements.

"""

import numpy as np
import scipy.optimize
from scipy.stats import norm 
from matplotlib  import pyplot as plt

class KDE_1d:
    def __init__(self, data):
        self.data = data
        self.setInterval()
        self.points = np.array([])
        self.bins   = np.array([])
        self.vals   = np.array([]) 
        self.step   = 0
    
    def setInterval(self):
        min = np.amin(self.data)
        max = np.amax(self.data)
        self.std = np.std(self.data)
        self.xmin = min-self.std
        self.xmax = max+self.std
    
    def getKDE(self, bandwidth = 0, gridfactor = 5):

        if bandwidth == 0:
            bandwidth = 1.06*self.std/(len(self.data)**(1.0/5))

        self.points = np.arange(self.xmin,self.xmax+bandwidth/gridfactor,bandwidth/gridfactor)
        self.vals = np.zeros(len(self.points))
        self.step   = bandwidth/gridfactor
        self.bins = self.points - self.step/2
        self.bins = np.append(self.bins, [self.bins[-1]+self.step])

        spread_factors = [norm.cdf(-5 + ((i+0.5)/gridfactor))-norm.cdf(-5 + ((i-0.5)/gridfactor)) for i in range(10*gridfactor+1)]

        self.vals = np.histogram(self.data,bins = self.bins)[0]

        spreaded = np.zeros(len(self.vals))
        for i in range(10*gridfactor+1):
            spreaded[i:-10*gridfactor+i-1] += spread_factors[i]*self.vals[5*gridfactor:-(5*gridfactor)-1] 
        self.vals = spreaded

        normalization = np.sum(self.vals)*bandwidth/gridfactor
        self.vals = self.vals/normalization
        
        return self.points, self.vals, self.step

    def getBins(self):
        return self.bins

    def objective( self, density, prob):
        mask = self.vals>density
        count = self.vals[mask]*self.step
        return count.sum() - prob


    def getPLMask( self, level = 0.9):
        prob = scipy.optimize.bisect(self.objective, self.vals.min(), self.vals.max(), args=(level,))
        mask = self.vals>prob
        return mask, prob
        
    def Plot(self, ax):
        x,y,w = self.getKDE()
        b     = self.getBins()
        M_68, prob_68d     = self.getPLMask(level=0.6827)
        M_95, prob_95d     = self.getPLMask(level=0.9545)
        M_99, prob_99d     = self.getPLMask(level=0.9973)
        y_68           = np.array([0 if not M_68[i] else y[i] for i in range(len(y)) ])
        y_95           = np.array([0 if not M_95[i] else y[i] for i in range(len(y)) ])
        y_99           = np.array([0 if not M_99[i] else y[i] for i in range(len(y)) ])
        mean_x=np.mean(self.data)
        ax.plot(x,y)
        ax.hist(b[:-1],b,weights=y_99,color='yellow')
        ax.hist(b[:-1],b,weights=y_95,color='orange')
        ax.hist(b[:-1],b,weights=y_68,color='red')
        ax.axvline(mean_x, color='b', linestyle='dashed')