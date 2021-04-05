""" This code is am example of a complete bayesian analysis.
    Data are taken from table 15.5 of Dekking, et.al, "A Modern Introduction to
    Probability and Statistics" """

import numpy as np
import matplotlib.pyplot as plt

print("Bayesian model of Density and Hardness of Australian Timber")

Data = np.array([ 
   [24.7,484],
   [24.8,427],
   [27.3,413],
   [28.4,517],
   [28.4,549],
   [29.0,648],
   [30.3,587],
   [32.7,704],
   [35.6,979],
   [38.5,914],
   [38.8,1070],
   [39.3,1020],
   [39.4,1210],
   [39.9,989],
   [40.3,1160],
   [40.6,1010],
   [40.7,1100],
   [40.7,1130],
   [42.9,1270],
   [45.8,1180],
   [46.9,1400],
   [48.2,1760],
   [51.5,1710],
   [51.5,2010],
   [53.4,1880],
   [56.0,1980],
   [56.5,1820],
   [57.3,2020],
   [57.6,1980],
   [59.2,2310],
   [59.8,1940],
   [66.0,3260],
   [67.4,2700],
   [68.8,2890],
   [69.1,2740],
   [69.1,3140]])
print("Data: ", Data)


# Defining data arrays
Data_x = Data[:,0]
Data_y = Data[:,1]
Data_yerr = np.array(len(Data_y)*[1])

# now least square fit
A = np.vander(Data_x, 2)
C = np.diag(Data_yerr * Data_yerr)
ATA = np.dot(A.T, A / (Data_yerr ** 2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, Data_y / Data_yerr ** 2))
m_ls = w[0]
b_ls = w[1]

print("Least-squares estimates:")
print("m = {0:.3f} ± {1:.3f}".format(w[0], np.sqrt(cov[0, 0])))
print("b = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))



# now plotting: scatter plot and residuals
x_min = min(Data_x)
x_max = max(Data_x)
x_0 = np.linspace(x_min,x_max,num=500)
fig, axs = plt.subplots(2)
fig.suptitle('Fitting wood hardness')
axs[0].scatter(Data_x,Data_y)
axs[0].plot(x_0,(m_ls*x_0+b_ls))
axs[1].scatter(Data_x,Data_y-(m_ls*Data_x+b_ls))
plt.show()
# plotting residual histograms
plt.hist(Data_y-(m_ls*Data_x+b_ls))
plt.show()




