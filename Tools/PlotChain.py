#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 07:42:20 2021

@author: P.Chimenti


  Utility class to make a simple plot of a MCMC
  
"""

import matplotlib.pyplot as plt 
import numpy as np

class PlotChain():
        def __init__(self):
            pass
        
        def Plot(self, data, name = "var1", file_tag = ""):
            fig, ax = plt.subplots()
            ax.plot( np.arange(start=1, stop=len(data)+1), data)
            ax.set(xlabel='iteration', ylabel=name,
                   title='Chain - '+name)
            ax.grid()        
            fig.savefig("Chain_"+name+"_"+file_tag+".png")
            plt.close()