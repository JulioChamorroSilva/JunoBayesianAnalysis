import numpy as np
import matplotlib.pyplot as plt

from Model import Commons as com
from Model import UnitsConstants as UC
from Model import SurvivalProbability_ME

plt.style.use('Styles/Paper.mplstyle')

DelM2_21 = (7.53e-5*UC.eV**2)/(UC.C**4)
DelM2_31 = (2.5e-3*UC.eV**2)/(UC.C**4)
Theta_12 = 33.82*UC.deg
Theta_13 = 8.61*UC.deg
prob     = SurvivalProbability_ME.SurvivalProbability_ME( DelM2_21,     DelM2_31, Theta_12, Theta_13, com.RockDensity )
prob_0   = SurvivalProbability_ME.SurvivalProbability_ME( DelM2_21,     DelM2_31, Theta_12, Theta_13, 0 )
prob_IH  = SurvivalProbability_ME.SurvivalProbability_ME( DelM2_21, -1.*DelM2_31, Theta_12, Theta_13, com.RockDensity )

L    = 52.*UC.km
E_nu = 1.*UC.MeV 
print("E=",E_nu/UC.MeV,"MeV,L=",L/UC.km,"km, matter effect", prob.Pee(  E_nu, L )) # E = 2 
print("E=",E_nu/UC.MeV,"MeV,L=",L/UC.km,"km, no matter effect", prob_0.Pee(E_nu, L))
print("E=",E_nu/UC.MeV,"MeV,L=",L/UC.km,"km, matter effect - IH", prob_IH.Pee(E_nu, L))

x_e      = np.linspace(1, 10, 10001)*UC.MeV
y_pee    = [ prob.Pee( i, L ) for i in x_e ]
y_pee_0  = [ prob_0.Pee( i, L ) for i in x_e ]
y_pee_IH = [ prob_IH.Pee( i, L ) for i in x_e ]
plt.plot( x_e/UC.MeV, y_pee, label='Normal ordering', linewidth='0.5'      )
plt.plot( x_e/UC.MeV, y_pee_IH, label='Inverted ordering', linewidth='0.5'   )
plt.xlabel("Energy (MeV)")
plt.ylabel("Probability")
plt.legend()
plt.title("Electron Neutrino Survival Probability")
plt.grid(linestyle='--')
plt.show()
plt.clf()

ratio_ME = [ (y_pee[i] - y_pee_0[i])/y_pee[i] for i in range(len(y_pee)) ]
ratio_IH = [ (y_pee_IH[i] - y_pee[i])/y_pee[i] for i in range(len(y_pee)) ]
plt.plot( x_e/UC.MeV, ratio_ME, label='P $_{normal-vacuum}$', linewidth='0.5', zorder=10 )
plt.plot( x_e/UC.MeV, ratio_IH, label='P $_{inverted-normal}$', linewidth='0.5', zorder=0  )
plt.xlabel("Energy (MeV)")
plt.ylabel("Ratio of probability")
plt.title("Ratio of probability over Normal ordering")
plt.legend()
plt.grid(linestyle='--')
plt.show()
