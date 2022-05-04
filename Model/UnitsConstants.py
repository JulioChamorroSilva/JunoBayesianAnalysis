#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:09:31 2021

@author: Pietro Chimenti

in this project we use MKS units
"""
import numpy as np

C    = 299792458            # light speed - m/s
Q_ep = 1.60217662E-19       # positron charge - Coulomb
Hbar = 1.05457168E-34       # Planck constant/2 Pi - J s
Av   = 6.02214076E23        # Avogadro mol^-1

eV   = Q_ep                 # electron volt em Joules
MeV  = 1e6*eV
GeV  = 1e9*eV

Gf   = 1.1663787E-5*(GeV**-2)   # Fermi constant over (hbar c)^3

M_p  = 938.27208816*MeV/C**2  # Proton mass
M_n  = 939.56542052*MeV/C**2  # Neutron mass

cm   = 1e-2                 # in m
km   = 1e3                  # in m

g    = 1e-3                 # g in Kg

deg  = np.pi/180.           # degree to radian
