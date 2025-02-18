#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 21:19:48 2021

This class describes the initial antineutrino spectrum according to huber-muller model
It is based on juno common inputs

@author: chimenti
"""
import uproot #uproot is the VSCode friendly version of ROOT necessary to smoothly run on Windows
from Model import UnitsConstants as UC
import math

class Spectrum_HM():
    def __init__(self, path, name):
        self.file_common = uproot.open(path,"READ")
        self.f_HuberMuellerFlux = self.file_common.Get(name) 
        self.nbins      = self.f_HuberMuellerFlux.GetNbinsX()
        self.low_limit  = self.f_HuberMuellerFlux.GetBinLowEdge( 1 )*UC.MeV 
        self.high_limit = self.f_HuberMuellerFlux.GetBinLowEdge( self.nbins+1 )*UC.MeV
        self.bin_width  = self.f_HuberMuellerFlux.GetBinWidth( 1 )*UC.MeV
        
    def Rate(self, Enu):
        bin = math.floor((Enu-self.low_limit)/self.bin_width)
        return self.f_HuberMuellerFlux.GetBinContent(bin)/UC.MeV
        
        
        
        
