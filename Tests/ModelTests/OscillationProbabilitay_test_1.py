import sys
sys.path.append("../..")

from Model.SurvivalProbability_ME import *

prob = SurvivalProbability_ME(1.2, 3.5, 1., 3., 1.)

print(prob.Pee(2.))


