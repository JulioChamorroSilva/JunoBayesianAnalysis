#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 22:19:42 2021

testing HM spectra

@author: chimenti
"""
from   Model import Spectrum_HM
from   Model import UnitsConstants as UC
import scipy.integrate as integ
import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer as timer

u235 = Spectrum_HM.Spectrum_HM("../juno-common/CommonInput/JUNOInputs2020_12_22.root","HuberMuellerFlux_U235")
u238 = Spectrum_HM.Spectrum_HM("../juno-common/CommonInput/JUNOInputs2020_12_22.root","HuberMuellerFlux_U238")
pu239 = Spectrum_HM.Spectrum_HM("../juno-common/CommonInput/JUNOInputs2020_12_22.root","HuberMuellerFlux_Pu239")
pu241 = Spectrum_HM.Spectrum_HM("../juno-common/CommonInput/JUNOInputs2020_12_22.root","HuberMuellerFlux_Pu241")

start = timer()
print(integ.quadrature(u235.Rate,u235.low_limit, u235.high_limit, rtol = 0.0001, vec_func = False))
end = timer()
print(end - start)

x_enu    = np.linspace( u235.low_limit, u235.high_limit, u235.nbins+1)
Spectrum_u235 = [u235.Rate(i)*UC.MeV for i in x_enu]
Spectrum_u238 = [u238.Rate(i)*UC.MeV for i in x_enu]
Spectrum_pu239 = [pu239.Rate(i)*UC.MeV for i in x_enu]
Spectrum_pu241 = [pu241.Rate(i)*UC.MeV for i in x_enu]

plt.plot( x_enu[5:-1]/UC.MeV, Spectrum_u235[5:-1], label = "U235" )
plt.plot( x_enu[5:-1]/UC.MeV, Spectrum_u238[5:-1], label = "U238" )
plt.plot( x_enu[5:-1]/UC.MeV, Spectrum_pu239[5:-1], label = "PU239" )
plt.plot( x_enu[5:-1]/UC.MeV, Spectrum_pu241[5:-1], label = "PU241" )
plt.legend()
plt.yscale('log')
plt.show() 

