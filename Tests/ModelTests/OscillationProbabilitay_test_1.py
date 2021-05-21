import sys
sys.path.append("../..")
import numpy as np

from Model.SurvivalProbability_ME import *

prob = SurvivalProbability_ME(1.2, 3.5, 1., 3., 1.)

print(prob.Pee(2.))

# Fazendo uma modificação teste 1
# Fazendo uma modificação teste 2
