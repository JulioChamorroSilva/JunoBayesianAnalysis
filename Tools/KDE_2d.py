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


class KDE_2d:
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.points_x = np.array([])
        self.points_y = np.array([])
        self.vals   = np.array([]) 
        self.step_x   = 0
        self.step_y   = 0
        self.bandwidth_x = 0
        self.bandwidth_y = 0

    def spreadFunc(self, x, y):
        factor_x = sum(1./(np.sqrt(2.*np.pi)*self.bandwidth_x)*np.exp(-np.power((x - self.data_x)/self.bandwidth_x, 2.)/2))
        factor_y = sum(1./(np.sqrt(2.*np.pi)*self.bandwidth_y)*np.exp(-np.power((y - self.data_y)/self.bandwidth_y, 2.)/2))
        return  factor_x*factor_y
    
    def getKDE(self, bandwidth_x = 0, bandwidth_y = 0, gridfactor = 5, xmin = 0, xmax = 0, ymin = 0, ymax = 0):
        min_x = np.amin(self.data_x)
        max_x = np.amax(self.data_x)
        min_y = np.amin(self.data_y)
        max_y = np.amax(self.data_y)
        std_x = np.std(self.data_x)
        std_y = np.std(self.data_y)
        if bandwidth_x == 0:
            bandwidth_x = 1.06*std_x/(len(self.data_x)**(1.0/5))
        if bandwidth_y == 0:
            bandwidth_y = 1.06*std_y/(len(self.data_y)**(1.0/5))
        self.bandwidth_x = bandwidth_x
        self.bandwidth_y = bandwidth_y

        if xmin == 0:
            xmin = min_x-std_x
        if xmax == 0:
            xmax = max_x+std_x

        if ymin == 0:
            ymin = min_y-std_y
        if ymax == 0:
            ymax = max_y+std_y

        points_x = np.arange(xmin,xmax,bandwidth_x/gridfactor)
        points_y = np.arange(ymin,ymax,bandwidth_y/gridfactor)
        xx, yy = np.meshgrid(points_x, points_y)
        vals = np.zeros(xx.shape)
        v_spreadFunc = np.vectorize(self.spreadFunc)
        vals = v_spreadFunc(xx,yy)

        norm = np.sum(vals)*(bandwidth_x/gridfactor)*(bandwidth_y/gridfactor)
        vals = vals/norm

        self.points_x = xx
        self.points_y = yy
        self.vals   = vals 
        self.step_x   = bandwidth_x/gridfactor
        self.step_y   = bandwidth_y/gridfactor
        return xx, yy, vals


    def objective( self, density, prob):
        mask = self.vals>density
        count = self.vals[mask]*self.step_x*self.step_y
        return count.sum() - prob


    def getPLMask( self, level = 0.9):
        prob = scipy.optimize.bisect(self.objective, self.vals.min(), self.vals.max(), args=(level,))
        mask = self.vals>prob
        return mask, prob
        
        