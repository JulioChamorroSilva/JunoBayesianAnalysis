#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 21:19:48 2021

This class describes the U235 spectrum according to huber-muller model
It is based on juno common inputs

@author: chimenti
"""
import ROOT
from Model import UnitsConstants as UC
import math

class Spectrum_U235_HM():
    def __init__(self, path, name):
        self.file_common = ROOT.TFile(path,"READ")
        self.f_HuberMuellerFlux_U235 = self.file_common.Get(name) 
        self.nbins      = self.f_HuberMuellerFlux_U235.GetNbinsX()
        self.low_limit  = self.f_HuberMuellerFlux_U235.GetBinLowEdge( 1 )*UC.MeV 
        self.high_limit = self.f_HuberMuellerFlux_U235.GetBinLowEdge( self.nbins+1 )*UC.MeV
        self.bin_width  = self.f_HuberMuellerFlux_U235.GetBinWidth( 1 )*UC.MeV
        
    def Rate(self, Enu):
        bin = math.floor((Enu-self.low_limit)/self.bin_width)
        return self.f_HuberMuellerFlux_U235.GetBinContent(bin)/UC.MeV
        
        
        
        