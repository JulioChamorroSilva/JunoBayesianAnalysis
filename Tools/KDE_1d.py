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


class KDE_1d:
    def __init__(self, data):
        self.data = data
        self.setInterval()
        self.points = np.array([])
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

        self.points = np.arange(self.xmin,self.xmax,bandwidth/gridfactor)
        self.vals = np.zeros(len(self.points))
        self.step   = bandwidth/gridfactor
        self.bins = self.points - self.step/2
        self.bins = np.append(self.bins, [self.bins[-1]+self.step])

        for i in range(len(self.vals)):
            self.vals[i] = sum(1./(np.sqrt(2.*np.pi)*bandwidth)*np.exp(-np.power((self.points[i] - self.data)/bandwidth, 2.)/2))

        norm = np.sum(self.vals)*bandwidth/gridfactor
        self.vals = self.vals/norm
        
        return self.points, self.vals, bandwidth/gridfactor

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
        
        