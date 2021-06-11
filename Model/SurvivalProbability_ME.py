import numpy as np
import Model.Commons as com
import Model.UnitsConstants as UC

class SurvivalProbability_ME:
    """ This class represents the electron anti-neutrino oscillation probability
    to be used in JUNO for precision oscillation measurements and mass hierarchy, 
    including matter effects """
    def __init__(self, *params):
            self._DelM2_21 = params[0]
            self._DelM2_31 = params[1]
            self._Theta_12 = params[2]
            self._Theta_13 = params[3]
            self._Rho       = params[4]

    def Pee( self, E, L):
        # missing matter effect
        #A        = -2. * np.sqrt(2) * com.Gf * self._Ne * E         
        # DelM2_32 = self._DelM2_31 - self._DelM2_21 # m_3^2 - m_2^2 = ( m_3^2 - m_1^2 ) - ( m_2^2 - m_1^2 )
        N_e      = self._Rho/(2*UC.M_p)
        A        = -1.*( 2*np.sqrt(2) * UC.Gf * N_e * E * UC.Hbar**3 / UC.C)
        DelM2_31_ME = self._DelM2_31*np.sqrt( (np.cos( 2*self._Theta_13 ) - A / self._DelM2_31)**2 + np.sin( 2*self._Theta_13 )**2)
        DelM2_21_ME = self._DelM2_21*np.sqrt( (np.cos( 2*self._Theta_12 ) - A / self._DelM2_31)**2 + np.sin( 2*self._Theta_12 )**2)
        DelM2_32_ME = DelM2_31_ME - DelM2_21_ME 
        Delta31  = DelM2_31_ME * L * UC.C**3 / ( 4 * E * UC.Hbar )
        Delta32  = DelM2_32_ME * L * UC.C**3 / ( 4 * E * UC.Hbar )
        Delta21  = DelM2_21_ME * L * UC.C**3 / ( 4 * E * UC.Hbar )
        term1   = np.sin( 2*self._Theta_13 )**2 * np.cos(   self._Theta_12 )**2 * np.sin( Delta31 )**2 
        term2   = np.sin( 2*self._Theta_13 )**2 * np.sin(   self._Theta_12 )**2 * np.sin( Delta32 )**2
        term3   = np.cos(   self._Theta_13 )**4 * np.sin( 2*self._Theta_12 )**2 * np.sin( Delta21 )**2
        return  1 - term1 - term2 - term3

