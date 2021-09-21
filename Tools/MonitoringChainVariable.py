#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 20:04:34 2021

@author: P.Chimenti

    This function implement the calculation of the potential scale reduction factor R 
    as well as the variogram and effective sample number
    (see. A.Gelman et al., "Bayesian Data Analysis" pg. 284)

"""

import numpy as np

class Monitor():
        def __init__(self, data, seq_num, discard_factor):
            self.data = data
            self.seq_num = seq_num
            self.discard_factor = discard_factor
            skip = int(round(len(data)*discard_factor))
            self.Process(data[skip:],seq_num)            

        def Process(self, data , seq_num):
            samples = len(data)
            seq_samples = samples//seq_num
            sequences = [ data[i*seq_samples:(i+1)*seq_samples] for i in range(seq_num)]
            psi_dot = np.array([sum(seq)/seq_samples for seq in sequences])
            psi_dot_dot = sum(psi_dot)/seq_num
            B = (seq_samples/(seq_num-1))*sum((psi_dot-psi_dot_dot)**2)
            s_square = np.array([ sum((p-pd)**2)/(seq_samples-1) for (p,pd) in zip(sequences,psi_dot)])
            W = sum(s_square)/seq_num
            var_plus = (((seq_samples-1)/seq_samples))*W+(B/seq_samples)
            R = np.sqrt(var_plus/W)

            self.seq_samples = seq_samples
            self.sequences = sequences
            self.var_plus = var_plus
            self.R = R
            
        def n_eff(self):
            seq_samples = self.seq_samples # just avoid using "self"
            sequences   = self.sequences
            seq_num     = self.seq_num
            var_plus    = self.var_plus

            print("n_eff function not yet tested!")
            variogram = []
            for t in range(1,seq_samples-1):
                Vector_t = np.array([sum((sequences[j][t+1:-1]-sequences[j][0:-(t+1)-1])**2)/(seq_samples-t) for j in range(len(sequences))])
                variogram.append(sum(Vector_t)/seq_num)
                
            V_t = np.array(variogram)
            rho_t = 1-V_t/(2*var_plus)
            rho_sum = np.array([sum(rho_t[:i]) for i in np.arange(1,len(rho_t-1))])
            n_eff = (seq_num*seq_samples)/(1+2*rho_sum)

            self.V_t = V_t
            self.n_eff = n_eff
    