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
        ''' All the equations for the effect of matter were added from 
        the article 'Why matter effects matter for JUNO' written 
        by Amir N. Khan, Hiroshi Nunokawa and Stephen J. Parke, 
        available at https://doi.org/10.1016/j.physletb.2020.135354 '''
        N_e      = self._Rho/(2*UC.M_p)         # This equation returns the Electron Number Density using the input Rho, the Rock Density, divided by two times the Proton Mass
        A        = -1.*( 2*np.sqrt(2) * UC.Gf * N_e * E * UC.Hbar**3 / UC.C)        # This is equation (5) in the article with the modifications necessary to convert the values back to S.I. units 
        DelM2_32 = self._DelM2_31 - self._DelM2_21      # This relation is described with equation (3) in the article
        DelM2_ee = np.cos(  self._Theta_12  )**2 * self._DelM2_31 + np.sin(  self._Theta_12  )**2 * DelM2_32        # This is equation (6) in the article
        DelM2_ee_ME = DelM2_ee * np.sqrt( (  np.cos(  2 * self._Theta_13  ) - (A / DelM2_ee) )**2 + np.sin(  2 * self._Theta_13  )**2 ) # This equation is equation (8) in the article
        A_ME = (A + DelM2_ee - DelM2_ee_ME)/2       # This is equation (11) in the article
        Cos2_theta_13_ME_minus_theta_13 = (DelM2_ee_ME + DelM2_ee - A * np.cos(2 * self._Theta_13)) / (2 * DelM2_ee_ME)         # This is equation (12) in the article
        DelM2_21_ME = self._DelM2_21 * np.sqrt( ( np.cos(2 * self._Theta_12) - A_ME/self._DelM2_21 )**2 + Cos2_theta_13_ME_minus_theta_13 * np.sin(2 * self._Theta_12)**2 )         # This is equation (10) in the article
        Cos2theta13_ME = (DelM2_ee * np.cos(2 * self._Theta_13) - A) / DelM2_ee_ME      # This is equation (7) in the article
        Theta_13_ME = np.arccos(Cos2theta13_ME)/2.      # This equation returns the value of Theta from the equation (7)
        Cos2theta12_ME = (self._DelM2_21 * np.cos(2 * self._Theta_12) - A_ME) / DelM2_21_ME         # This is equation (9) in the article
        Theta_12_ME = np.arccos(Cos2theta12_ME)/2.      # This equation returns the value of Theta from the equation (9)
        DelM2_31_ME = DelM2_ee_ME + (np.sin( Theta_12_ME )**2) * DelM2_21_ME        # This is equation (13) in the article
        DelM2_32_ME = DelM2_31_ME - DelM2_21_ME         # This is equation (14) in the article
        ''' These next equations return Delta from the DelM variables 
        as described by the article in equation (4) , then later use the 
        values of Delta in the equation (4) '''
        Delta21_ME = (DelM2_21_ME * L * UC.C**3 ) / (4 * E * UC.Hbar )  
        Delta32_ME = (DelM2_32_ME * L * UC.C**3 ) / (4 * E * UC.Hbar ) 
        Delta31_ME = (DelM2_31_ME * L * UC.C**3 ) / (4 * E * UC.Hbar ) 
        term1   = np.sin( 2*Theta_13_ME )**2 * np.cos(   Theta_12_ME )**2 * np.sin( Delta31_ME )**2 
        term2   = np.sin( 2*Theta_13_ME )**2 * np.sin(   Theta_12_ME )**2 * np.sin( Delta32_ME )**2
        term3   = np.cos(   Theta_13_ME )**4 * np.sin( 2*Theta_12_ME )**2 * np.sin( Delta21_ME )**2
        return  1. - term1 - term2 - term3      # This is equation (4) in the article
