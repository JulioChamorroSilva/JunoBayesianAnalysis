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
        # equations for the effect of matter added
        # all relevant variables are isolated so they can be calculated and input on the final equations
        N_e      = self._Rho/(2*UC.M_p)
        A        = -1.*( 2*np.sqrt(2) * UC.Gf * N_e * E * UC.Hbar**3 ) #multiplying by Hbar^3 is necessary to return the variable to natural units 
        DelM2_32 = self._DelM2_31 - self._DelM2_21
        DelM2_ee = np.cos(  self._Theta_12  )**2 * self._DelM2_31 + np.sin(  self._Theta_12  )**2 * DelM2_32
        DelM2_ee_ME = DelM2_ee * np.sqrt(  np.cos(  2 * self._Theta_13  )**2 + np.sin(  2 * self._Theta_13  ) )
        A_ME = (A + DelM2_ee - DelM2_ee_ME)/2 
        DelM2_21_ME = self._DelM2_21 * np.sqrt( ( np.cos(2 * self._Theta_12) - A_ME/self._DelM2_21 )**2 + ( DelM2_ee_ME + DelM2_ee - A * np.cos( self._Theta_13 ))/2 * np.sin(self._Theta_12)**2 )
        Theta_13_ME = 1 / (2 * np.cos((DelM2_ee * np.cos(2 * self._Theta_13) - A) / DelM2_ee_ME))
        Theta_12_ME = 1 / (2 * np.cos((self._DelM2_21 * np.cos(self._Theta_13) - A_ME) / DelM2_21_ME))
        DelM2_31_ME = DelM2_ee_ME + (np.sin( Theta_12_ME )**2) * DelM2_21_ME 
        DelM2_32_ME = DelM2_31_ME - DelM2_21_ME
        Delta21_ME = np.sqrt(DelM2_21_ME) * L * UC.Hbar / (4 * E * UC.C**3 * 1e15 ) # Delta has dimentional unit of eV^2/c * Km * GeV^-1, so to to convert GeV and eV back to natural units we have to multiply by Hbar / 10^15 * c^3
        Delta32_ME = np.sqrt(DelM2_32_ME) * L * UC.Hbar / (4 * E * UC.C**3 * 1e15 ) # Delta has dimentional unit of eV^2/c * Km * GeV^-1, so to to convert GeV and eV back to natural units we have to multiply by Hbar / 10^15 * c^3
        Delta31_ME = np.sqrt(DelM2_31_ME) * L * UC.Hbar / (4 * E * UC.C**3 * 1e15 ) # Delta has dimentional unit of eV^2/c * Km * GeV^-1, so to to convert GeV and eV back to natural units we have to multiply by Hbar / 10^15 * c^3
        term1   = np.sin( 2*Theta_13_ME )**2 * np.cos(   Theta_12_ME )**2 * np.sin( Delta31_ME )**2 
        term2   = np.sin( 2*Theta_13_ME )**2 * np.sin(   Theta_12_ME )**2 * np.sin( Delta32_ME )**2
        term3   = np.cos(   Theta_13_ME )**4 * np.sin( 2*Theta_12_ME )**2 * np.sin( Delta21_ME )**2
        return  1 - term1 - term2 - term3


