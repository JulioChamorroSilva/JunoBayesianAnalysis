import emcee
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

J = 8
y_obs = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

def log_prior_8school(theta):
     mu, tau, eta = theta[0], theta[1], theta[2:]
     # Half-cauchy prior, hwhm=25
     if tau < 0:
         return -np.inf
     prior_tau = -np.log(tau ** 2 + 25 ** 2)
     prior_mu = -(mu / 10) ** 2  # normal prior, loc=0, scale=10
     prior_eta = -np.sum(eta ** 2)  # normal prior, loc=0, scale=1
     return prior_mu + prior_tau + prior_eta

def log_likelihood_8school(theta, y, s):
     mu, tau, eta = theta[0], theta[1], theta[2:]
     return -((mu + tau * eta - y) / s) ** 2

def lnprob_8school(theta, y, s):
     prior = log_prior_8school(theta)
     like_vect = log_likelihood_8school(theta, y, s)
     like = np.sum(like_vect)
     return like + prior

nwalkers, draws = 50, 700
ndim = J + 2
pos = np.random.normal(size=(nwalkers, ndim))
pos[:, 1] = np.absolute(pos[:, 1])

sampler = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    lnprob_8school,
    args=(y_obs, sigma)
)

sampler.run_mcmc(pos, draws);
var_names = ['mu', 'tau']+['eta{}'.format(i) for i in range(J)]
emcee_data = az.from_emcee(sampler, var_names=var_names).sel(draw=slice(100, None))
az.plot_posterior(emcee_data, var_names=var_names[:3])
#plt.show()
plt.savefig('test_arviz_fig1.svg', format='svg', dpi=1200)
plt.close()

emcee_data = az.from_emcee(sampler, slices=[0, 1, slice(2, None)])
az.plot_trace(emcee_data, var_names=["var_2"], coords={"var_2_dim_0": 4})
#plt.show()
plt.savefig('test_arviz_fig2.svg', format='svg', dpi=1200)
plt.close()

def lnprob_8school_blobs(theta, y, s):
    prior = log_prior_8school(theta)
    like_vect = log_likelihood_8school(theta, y, s)
    like = np.sum(like_vect)
    return like + prior, like_vect

sampler_blobs = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    lnprob_8school_blobs,
    args=(y_obs, sigma)
)

sampler_blobs.run_mcmc(pos, draws);
dims = {"eta": ["school"], "log_likelihood": ["school"]}
data = az.from_emcee(
    sampler_blobs,
    var_names = ["mu", "tau", "eta"],
    slices=[0, 1, slice(2,None)],
    blob_names=["log_likelihood"],
    dims=dims,
    coords={"school": range(8)}
)

def lnprob_8school_blobs(theta, y, sigma):
    mu, tau, eta = theta[0], theta[1], theta[2:]
    prior = log_prior_8school(theta)
    like_vect = log_likelihood_8school(theta, y, sigma)
    like = np.sum(like_vect)
    return like + prior, (like_vect, np.random.normal((mu + tau * eta), sigma))

sampler_blobs = emcee.EnsembleSampler(
    nwalkers,
    ndim,
    lnprob_8school_blobs,
    args=(y_obs, sigma),
)

sampler_blobs.run_mcmc(pos, draws);
dims = {"eta": ["school"], "log_likelihood": ["school"], "y": ["school"]}
data = az.from_emcee(
    sampler_blobs,
    var_names = ["mu", "tau", "eta"],
    slices=[0, 1, slice(2,None)],
    arg_names=["y","sigma"],
    arg_groups=["observed_data", "constant_data"],
    blob_names=["log_likelihood", "y"],
    blob_groups=["log_likelihood", "posterior_predictive"],
    dims=dims,
    coords={"school": range(8)}
)
az.plot_ppc(data, var_names=["y"], alpha=0.3, num_pp_samples=50)
#plt.show()
plt.savefig('test_arviz_fig3.svg', format='svg', dpi=1200)
plt.close()


