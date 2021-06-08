import numpy as np
import matplotlib.pyplot as plt

from Model import Commons as com
from Model import UnitsConstants as UC
from Model import SurvivalProbability_ME

DelM2_21 = 7.53e-5*UC.eV**2/UC.C**4
DelM2_31 = 2.5e-3*UC.eV**2/UC.C**4
Theta_12 = 33.82*UC.deg
Theta_13 = 8.61*UC.deg
prob = SurvivalProbability_ME.SurvivalProbability_ME( DelM2_21, DelM2_31, Theta_12, Theta_13, 0 )

L    = 52.*UC.km
E_nu = 1.*UC.MeV 
print(prob.Pee(  E_nu, L )) # E = 2 

x_e = np.linspace(1, 10, 1001)*UC.MeV
y_pee = [ prob.Pee( i, L ) for i in x_e ]
plt.plot( x_e, y_pee )
plt.show()

