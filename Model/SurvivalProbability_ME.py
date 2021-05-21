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

theta_13 = 1
theta_12 = 1
delta_31 = 1
delta_32 = 1
delta_21 = 1

p = 1 - (((np.sin(2*theta_13))**2)*(((np.cos(theta_12))**2)\
*((np.sin(delta_31))**2)+ ((np.sin(theta_12))**2)*((np.sin(delta_32))**2)))-\
((np.cos(theta_13))**4)*((np.sin(2*theta_12))**2)*((np.sin(delta_21))**2)

# delta_ij= ((mi**2 - mj**2)*L)/(4*E)

# (delta_mij)**m = np.sqrt((np.cos(2*theta_ij-A/delta_mij)**2)+(np.sin(2*theta_ij)**2))

print(p)