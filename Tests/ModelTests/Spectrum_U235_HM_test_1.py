#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 22:19:42 2021

testing U235 spectrum

@author: chimenti
"""
from   Model import Spectrum_U235_HM
from   Model import UnitsConstants as UC
import scipy.integrate as integ
import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer as timer

u235 = Spectrum_U235_HM.Spectrum_U235_HM("../juno-common/CommonInput/JUNOInputs2020_12_22.root","HuberMuellerFlux_U235")
start = timer()
print(integ.quadrature(u235.Rate,u235.low_limit, u235.high_limit, rtol = 0.0001, vec_func = False))
end = timer()
print(end - start)

x_enu    = np.linspace( u235.low_limit, u235.high_limit, u235.nbins+1)
Spectrum = [u235.Rate(i)*UC.MeV for i in x_enu] 
plt.plot( x_enu[5:-1]/UC.MeV, Spectrum[5:-1])
plt.yscale('log')
plt.show() 

