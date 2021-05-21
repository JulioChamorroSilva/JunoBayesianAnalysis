import numpy as np
import matplotlib.pyplot as plt

from   Model import Commons as com
from Model.SurvivalProbability_ME import *

prob = SurvivalProbability_ME(1.2, 3.5, 1., 3., com.Ne )

L = 52.
print(prob.Pee(  2., L )) # E = 2 

x_e = np.linspace(1, 10, 1001)
print(x_e)
y_pee = [ prob.Pee( i, L ) for i in x_e ]
plt.plot( x_e, y_pee )
plt.show()

