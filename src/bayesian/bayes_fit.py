from pymcmcstat.MCMC import MCMC
import numpy as np
from abc import ABC, abstractmethod

class AbstractBayesFitter(ABC):
    """The Product interface declares the operations that all concrete products must implement."""
    @abstractmethod
    def fit(self, *args, **kwargs) -> dict:
        pass

class AbstractMarginalLikelihoodEstimator(ABC):
    """The Product interface declares the operations that all concrete products must implement."""
    @abstractmethod
    def get_ml(self) -> dict:
        pass

def bf_from_2_log_marginal_lihelihoods(ml1, ml2, factor=100):
    return (np.exp(ml1 / factor) / np.exp(ml2 / factor)) ** factor

def HDI_from_MCMC(posterior_samples, credible_mass=0.95):
    if len(posterior_samples) == 0 or not (0 < credible_mass < 1):
        raise ValueError("Invalid posterior samples or credible mass.")
    sorted_points = sorted(posterior_samples)
    ciIdxInc = np.ceil(credible_mass * len(sorted_points)).astype('int')
    nCIs = len(sorted_points) - ciIdxInc
    ciWidth = [sorted_points[i + ciIdxInc] - sorted_points[i] for i in range(nCIs)]
    HDImin = sorted_points[ciWidth.index(min(ciWidth))]
    HDImax = sorted_points[ciWidth.index(min(ciWidth)) + ciIdxInc]
    return [HDImin, HDImax]

def get_trajectories(chain, model, t):
    if len(chain.shape) != 2 or chain.shape[1] != len(t):
        raise ValueError("Invalid dimensions for chain or time vector.")
    chain_len = len(chain)
    ys_simul = np.zeros((chain_len, len(t)))
    for k in range(chain_len):
        ys_simul[k, :] = model.simulate_theta(t, chain[k, :])[:len(t)]
    return ys_simul

def get_bound_trajectories(all_traj, credible_mass=0.95):
    y_mean = np.mean(all_traj, axis=0)
    ys_simul_T = np.transpose(all_traj)
    y_up, y_dw = [], []
    for day_samples in ys_simul_T:
        hdi_min, hdi_max = HDI_from_MCMC(day_samples, credible_mass)
        y_dw.append(hdi_min)
        y_up.append(hdi_max)
    return [np.array(y_dw), np.array(y_up), y_mean]

def get_dic(chain, LH_model):
    if chain.size == 0:
        raise ValueError("Chain is empty.")
    chain_T = np.transpose(chain)
    log_likelihoods = np.array([-2 * LH_model.get_logLH(p) for p in chain])
    theta_mean = np.mean(chain_T, axis=1)
    d_mean = np.mean(log_likelihoods)
    d_theta_mean = -2 * LH_model.get_logLH(theta_mean)
    return d_mean + (d_mean - d_theta_mean)

class DramFitter(AbstractBayesFitter):
    def __init__(self, fit_dict):
        self.prior = fit_dict.get('prior', [])
        self.theta_limits = fit_dict['theta_limits']
        self.init_theta = fit_dict['init_theta']
        self.theta_names = fit_dict['theta_names']
        self.LH_model = fit_dict['LH_model']
        self.maxiter = fit_dict['maxiter']
        self.b = fit_dict.get('b', 1)
        self.results = {}

    def log_prior(self, theta):
        p_prior_vector = [prior_fn(th) for prior_fn, th in zip(self.prior, theta)]
        return np.log(np.prod([max(p, 1e-9) for p in p_prior_vector]))

    def fit(self, temp=1):
        mcstat = MCMC()
        mcstat.data.add_data_set(self.LH_model.t, self.LH_model.data)

        def test_ssfun(theta, data):
            b_likelihood = np.exp(self.LH_model.get_logLH(theta)) ** self.b
            return -2 * (temp * np.log(b_likelihood) + self.log_prior(theta))

        mcstat.simulation_options.define_simulation_options(nsimu=self.maxiter, updatesigma=True)
        mcstat.model_settings.define_model_settings(sos_function=test_ssfun)

        for i, theta_0 in enumerate(self.init_theta):
            mcstat.parameters.add_model_parameter(
                name=self.theta_names[i],
                theta0=theta_0,
                minimum=self.theta_limits[i][0],
                maximum=self.theta_limits[i][1],
            )

        mcstat.run_simulation()
        self.results['dram_results'] = mcstat.simulation_results.results
        return self.results
