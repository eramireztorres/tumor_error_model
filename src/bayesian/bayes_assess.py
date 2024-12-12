import numpy as np
from scipy import stats
from abc import ABC, abstractmethod


class AbstractBayesFitter(ABC):
    """Interface for Bayesian Fitters."""

    @abstractmethod
    def fit(self, *args, **kwargs) -> dict:
        pass


class AbstractMarginalLikelihoodEstimator(ABC):
    """Interface for Marginal Likelihood Estimators."""

    @abstractmethod
    def get_ml(self) -> dict:
        pass


class TrajectoryPEstimator:
    @staticmethod
    def get_all_ps(data, all_traj, correct_vector=lambda x: 1, test="norm", stud_k=4):
        if not callable(correct_vector):
            raise ValueError("correct_vector must be a callable function.")
        ps = []
        for k, y_sim in enumerate(all_traj):
            y_correct = (data - y_sim) / correct_vector(k)
            if test == "t":
                k_result = stats.kstest(y_correct, "t", args=(stud_k,))
            else:
                k_result = stats.kstest(y_correct, test)
            ps.append(k_result.pvalue)
        return ps


class DramTiMarginalLikelihoodEstimator(AbstractMarginalLikelihoodEstimator):
    def __init__(self, fit_dict: dict, dram_fitter: AbstractBayesFitter):
        self.dram_fitter = dram_fitter
        self.n_temperatures = fit_dict["n_temperatures"]
        self.ti_N = fit_dict["ti_N"]
        self.results = {}

    def get_ml(self) -> dict:
        temperature_vector = np.linspace(0.1, 1, self.n_temperatures)
        log_posterior_likelihoods = []

        print("Init thermodynamic integration...")

        for i, temp in enumerate(temperature_vector ** self.ti_N):
            try:
                self.dram_fitter.fit(temp=temp)
                dram_results = self.dram_fitter.results.get("dram_results", {})
                chain = dram_results.get("chain", [])

                if len(chain) == 0:
                    raise ValueError("Chain is empty after fitting.")

                chain = chain[int(len(chain) / 2):]
                log_posterior = [self.dram_fitter.LH_model.get_logLH(p) for p in chain]
                log_posterior_likelihoods.append(np.array(log_posterior))

                print(f"Finished {i + 1} temperature of {self.n_temperatures}")
            except Exception as e:
                print(f"Error during temperature {i + 1}: {e}")
                continue

        if len(log_posterior_likelihoods) == 0:
            raise RuntimeError("No valid posterior likelihoods were computed.")

        try:
            log_marginal_likelihood = np.trapz(
                [np.mean(logs) for logs in log_posterior_likelihoods], temperature_vector
            )
        except Exception as e:
            raise RuntimeError(f"Error in log marginal likelihood computation: {e}")

        self.results["log_posterior_likelihoods"] = log_posterior_likelihoods
        self.results["log_marginal_likelihood"] = log_marginal_likelihood
        return self.results

