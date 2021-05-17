class SurvivalProbability_ME:
    """ This class represents the electron anti-neutrino oscillation probability
    to be used in JUNO for precision oscillation measurements and mass hierarqui, 
    including matter effects """
    def __init__(self, *params):
            self._DelM2_21 = params[0]
            self._DelM2_31 = params[1]
            self._Theta_12 = params[2]
            self._Theta_13 = params[3]
            self._Sign     = params[4]
        
    def Pee( self, E):
        return 1.

import numpy as np

theta13 = 1
theta12 = 1
delta31 = 1
delta32 = 1
delta21 = 1

p = 1 - (((np.sin(2*theta13))**2)*(((np.cos(theta12))**2)\
*((np.sin(delta31))**2)+ ((np.sin(theta12))**2)*((np.sin(delta32))**2)))-\
((np.cos(theta13))**4)*((np.sin(2*theta12))**2)*((np.sin(delta21))**2)

print(p)