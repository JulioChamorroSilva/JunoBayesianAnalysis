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
        self.points = np.array([])
        self.vals   = np.array([]) 
        self.step   = 0
    
    def getKDE(self, bandwidth = 0, gridfactor = 5, xmin = 0, xmax = 0):
        min = np.amin(self.data)
        max = np.amax(self.data)
        std = np.std(self.data)
        if bandwidth == 0:
            bandwidth = 1.06*std/(len(self.data)**(1.0/5))
        if xmin == 0:
            xmin = min-std
        if xmax == 0:
            xmax = max+std

        points = np.arange(xmin,xmax,bandwidth/gridfactor)
        vals = np.zeros(len(points))
        for i in range(len(vals)):
            vals[i] = sum(1./(np.sqrt(2.*np.pi)*bandwidth)*np.exp(-np.power((points[i] - self.data)/bandwidth, 2.)/2))

        norm = np.sum(vals)*bandwidth/gridfactor
        vals = vals/norm

        self.points = points
        self.vals   = vals 
        self.step   = bandwidth/gridfactor
        
        return points, vals, bandwidth/gridfactor

    def getBins(self):
        bins = self.points - self.step/2
        bins = np.append(bins, [bins[-1]+self.step])
        return bins

    def objective( self, density, prob):
        mask = self.vals>density
        count = self.vals[mask]*self.step
        return count.sum() - prob


    def getPLMask( self, level = 0.9):
        prob = scipy.optimize.bisect(self.objective, self.vals.min(), self.vals.max(), args=(level,))
        mask = self.vals>prob
        return mask, prob
        
        