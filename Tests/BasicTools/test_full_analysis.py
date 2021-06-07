""" This code is am example of a complete bayesian analysis.
    Data are taken from table 15.5 of Dekking, et.al, "A Modern Introduction to
    Probability and Statistics" """

import numpy as np
import matplotlib.pyplot as plt
import emcee
from scipy.optimize import minimize
import arviz as az

plt.style.use('Styles/Paper.mplstyle')

NSAMPLES = 10000
np.random.seed(20210426)
plot = True

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
Data_yerr = np.array(len(Data_y)*[1]) # just silly errors for LSF

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

if plot : 
    # now plotting: scatter plot and residuals
    x_min = min(Data_x)
    x_max = max(Data_x)
    x_0 = np.linspace(x_min,x_max,num=500)
    fig, axs = plt.subplots(2)
    fig.suptitle('Fitting wood hardness')
    axs[0].scatter(Data_x,Data_y)
    axs[0].plot(x_0,(m_ls*x_0+b_ls))
    axs[1].scatter(Data_x,Data_y-(m_ls*Data_x+b_ls))
    #plt.show()
    plt.savefig('test_full_analysis_fig1.svg', format='svg', dpi=1200)
    plt.close()
    # plotting residual histograms
    plt.hist(Data_y-(m_ls*Data_x+b_ls))
    #plt.show()
    plt.savefig('test_full_analysis_fig2.svg', format='svg', dpi=1200)
    plt.close()

# now bayesian

def log_likelihood(theta, x, y):
    m, b, sigma   = theta      # we need 3 parameters! The error is unknown
    model         = m * x + b
    return -0.5 * np.sum((y - model) ** 2 / (sigma**2) ) - len(x)*np.log(sigma) - len(x) * np.log(np.sqrt(2 * np.pi))


MAX_M = 1000
MAX_B = 10000
MAX_S = 1000

def log_prior(theta):
    m, b, sigma = theta
    if -1.*MAX_M < m < MAX_M and -1.*MAX_B < b < MAX_B and 0. < sigma < MAX_S:
        return 0.0
    return -np.inf

ppc = np.zeros(len(Data_x)+2) # blobs: prior,  ll, posterior predictive check -  simulated data

def log_probability(theta, x, y):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf, ppc
    # now calculate random samples
    for i in range(len(Data_x)):
        ppc[i+2] = theta[0] * Data_x[i] + theta[1] + np.random.normal(0., theta[2])
    ll = log_likelihood(theta, x, y)
    ppc[0] = lp
    ppc[1] = ll
    return lp + ll, ppc

# now run MCMC:

# sampling posterior

# first find maximum likelihood
nll = lambda *args: -log_likelihood(*args)
initial = np.array([50, -1000, 20]) # initial values from lest squared
sol_ML_linear = minimize(nll, initial, args=(Data_x, Data_y))
print("solution max. likelihood: ", sol_ML_linear)


# now sample bayesian distribution
pos = sol_ML_linear.x + 1e-4 * np.random.randn(32, 3)
nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(Data_x, Data_y))
sampler.run_mcmc(pos, NSAMPLES, progress=True);
flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
blobs        = sampler.get_blobs(discard=100, thin=15, flat=True)
inds = np.random.randint(len(flat_samples), size=100)

if plot :
    # now plotting
    # now arviz plots
    var_names = ['m','b','s']
    emcee_data = az.from_emcee(sampler, var_names=var_names, blob_names=["silly"] )
    az.plot_posterior(emcee_data, var_names=var_names[:])
    #plt.show()
    plt.savefig('test_full_analysis_fig3.svg', format='svg', dpi=1200)
    plt.close()

    # now trace plot
    az.plot_trace(emcee_data, var_names=var_names)
    #plt.show()
    plt.savefig('test_full_analysis_fig4.svg', format='svg', dpi=1200)
    plt.close()
    
    az.plot_pair(emcee_data, var_names=var_names, kind='kde', marginals=True)
    #plt.show()
    plt.savefig('test_full_analysis_fig5.svg', format='svg', dpi=1200)
    plt.close()

    print(flat_samples)
    print(blobs)
    print(blobs[0,:])
    print(blobs[:,0]) # this is the ppd of the first data

    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        sample = flat_samples[ind]
        loc_m = sample[0]
        loc_b = sample[1]
        plt.plot(x_0, loc_m * x_0 + loc_b, "C1", alpha=0.1)
        
    plt.scatter(Data_x,Data_y)
    plt.legend(fontsize=14)
    plt.xlabel("x")
    plt.ylabel("y");
    #plt.show()
    plt.savefig('test_full_analysis_fig6.svg', format='svg', dpi=1200)
    plt.close()

    for i in range(len(Data_x)):
        plt.hist(blobs[:,i+2], bins=20 ) # i+2: the first two blobs are prior and ll
        plt.axvline(Data_y[i], color='r')
        #plt.show()
        plt.savefig('test_full_analysis_fig7_'+str(i)+'.svg', format='svg', dpi=1200)
        plt.close()

print("--------------------------------")
#  now fit with fractional error
# fe : fractional error

MAX_S_FE = 1.

def log_likelihood_fe(theta, x, y):
    m, b, sigma_fe   = theta      # we need 3 parameters! The error is unknown
    model         = m * x + b
    return -0.5 * np.sum((y - model) ** 2 / (sigma_fe*y)**2 ) - np.sum(np.log(np.abs(sigma_fe*y))) - len(x) * np.log(np.sqrt(2 * np.pi))

def log_prior_fe(theta):
    m, b, sigma_fe = theta
    if -1.*MAX_M < m < MAX_M and -1.*MAX_B < b < MAX_B and 0. < sigma_fe < MAX_S_FE:
        return 0.0
    return -np.inf

def log_probability_fe(theta, x, y):
    lp = log_prior_fe(theta)
    if not np.isfinite(lp):
        return -np.inf, ppc
    # now calculate random samples
    for i in range(len(Data_x)):
        ppc[i+2] = theta[0] * Data_x[i] + theta[1] + np.random.normal(0., theta[2]*Data_y[i])
    ll = log_likelihood_fe(theta, x, y)
    ppc[0] = lp
    ppc[1] = ll
    return lp + ll, ppc

# first find maximum likelihood
nll_fe = lambda *args: -log_likelihood_fe(*args)
initial = np.array([50, -874, 0.1]) # initial values from lest squared
sol_ML_linear_fe = minimize(nll_fe, initial, args=(Data_x, Data_y))
print("solution max. likelihood second model: ", sol_ML_linear_fe)

# now sample bayesian distribution
pos = sol_ML_linear_fe.x + 1e-4 * np.random.randn(32, 3)
nwalkers, ndim = pos.shape
sampler_fe = emcee.EnsembleSampler(nwalkers, ndim, log_probability_fe, args=(Data_x, Data_y))
sampler_fe.run_mcmc(pos, NSAMPLES, progress=True);
flat_samples_fe = sampler_fe.get_chain(discard=100, thin=15, flat=True)
blobs_fe        = sampler_fe.get_blobs(discard=100, thin=15, flat=True)
inds_fe = np.random.randint(len(flat_samples), size=100)

if plot :
    # now plotting
    # now arviz plots
    var_names = ['m','b','s_fe']
    emcee_data_fe = az.from_emcee(sampler_fe, var_names=var_names, blob_names=["silly"] )
    az.plot_posterior(emcee_data_fe, var_names=var_names[:])
    plt.savefig('test_full_analysis_fig3_fe.svg', format='svg', dpi=1200)
    plt.close()
    
    # now trace plot
    az.plot_trace(emcee_data_fe, var_names=var_names)
    plt.savefig('test_full_analysis_fig4_fe.svg', format='svg', dpi=1200)
    plt.close()
    
    az.plot_pair(emcee_data_fe, var_names=var_names, kind='kde', marginals=True)
    plt.savefig('test_full_analysis_fig5.svg', format='svg', dpi=1200)
    plt.close()
    
    print(flat_samples_fe)
    
    inds = np.random.randint(len(flat_samples_fe), size=100)
    for ind in inds:
        sample_fe = flat_samples_fe[ind]
        loc_m = sample_fe[0]
        loc_b = sample_fe[1]
        plt.plot(x_0, loc_m * x_0 + loc_b, "C1", alpha=0.1)

    plt.scatter(Data_x,Data_y)
    plt.legend(fontsize=14)
    plt.xlabel("x")
    plt.ylabel("y");
    #plt.show()
    plt.savefig('test_full_analysis_fig6_fe.svg', format='svg', dpi=1200)
    plt.close()

    for i in range(len(Data_x)):
        plt.hist(blobs_fe[:,i+2], bins=20 )
        plt.axvline(Data_y[i], color='r')
        #plt.show()
        plt.savefig('test_full_analysis_fig7_'+str(i)+'_fe.svg', format='svg', dpi=1200)
        plt.close()
        
# now we compare models
# first AIC

print("--------------------------------")    
print("First model:")
print("ML estimate:", sol_ML_linear.x)
print("K = 3")
print("AIC = -2*p(d|t)+2*k = ",-2.*(log_likelihood(sol_ML_linear.x,Data_x, Data_y)),"+",2*3)
print("Second model:")
print("ML estimate:",sol_ML_linear_fe.x)
print("K = 3")
print("AIC = -2*p(d|t)+2*k = ",-2.*(log_likelihood_fe(sol_ML_linear_fe.x,Data_x, Data_y)),"+",2*3)

print("--------------------------------")
print("samples model 1:")
print(flat_samples)
print("blobs model 1:")
print(blobs)
mean_values = np.mean(flat_samples,axis = 0)
mean_blobs  = np.mean(blobs, axis = 0)
ll_bayes = log_likelihood(mean_values, Data_x, Data_y)
print("mean parameters:", mean_values)
print("ll_bayes model 1:", ll_bayes)
print("mean ll model 1:",mean_blobs[1])
elpd_DIC_1 = ll_bayes - 2*(ll_bayes-mean_blobs[1])
elpd_DIC_2 = ll_bayes - 2*np.var(blobs, axis = 0)[1]
print("Mean params model 1:",2*(ll_bayes-mean_blobs[1]),2*np.var(blobs, axis = 0)[1])
print("model 1 DIC:")
print(-2.*elpd_DIC_1, -2.*elpd_DIC_2)
print("--------------------------------")
print("samples model 2:")
print(flat_samples_fe)
print("blobs model 2:")
print(blobs_fe)
mean_values_fe = np.mean(flat_samples_fe,axis = 0)
mean_blobs_fe  = np.mean(blobs_fe, axis=0 )
ll_bayes_fe = log_likelihood_fe(mean_values_fe, Data_x, Data_y)
print("mean parameters model 2:", mean_values_fe)
print("ll_bayes model 2:", ll_bayes_fe)
print("mean ll model 2:", mean_blobs_fe[1])
elpd_DIC_1_fe = ll_bayes_fe - 2*(ll_bayes_fe-mean_blobs_fe[1])
elpd_DIC_2_fe = ll_bayes_fe - 2*np.var(blobs_fe, axis = 0)[1]
print("Mean params model 2:",2*(ll_bayes_fe-mean_blobs_fe[1]),2*np.var(blobs_fe, axis = 0)[1])
print("model 2 DIC:")
print(-2.*elpd_DIC_1_fe, -2.*elpd_DIC_2_fe)
print(">>>>>>>>>>   Missing autocorrelation!   <<<<<<<<<<<<<<<<<")
