from scipy.optimize import dual_annealing, basinhopping
import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import r2_score
from scipy import stats

class AbstractMlcFitter(ABC):
    """
    The Product interface declares the operations that all concrete products
    must implement.
    """

    @abstractmethod
    def fit(self) -> dict:
        pass
    
    def bic(self) -> dict:    
        #k: no. of parameters
        #N: sample size
        k = len(self.init_theta)
        N = len(self.LH_model.t)
        log_mle_value = self.results['log_mle_value']
        bic=k*np.log(N)-2*log_mle_value
        self.results['bic'] = bic
        return self.results

    def r2(self) -> dict:   
        mle_simulated_data = self.results['mle_simulated_data']
        data = self.LH_model.data
        r2 = r2_score(data, mle_simulated_data)
        self.results['R2'] = r2
        return self.results
    
    def shapiro(self) -> dict:   
        mle_simulated_data = self.results['mle_simulated_data']
        mle_theta = self.results['mle_theta']
        data = self.LH_model.data
        resid = mle_simulated_data - data
        correct_vector = self.LH_model.get_residual_correct_vector(mle_theta)
        shapiro_results = stats.shapiro(resid/correct_vector)
        self.results['shapiro_results'] = shapiro_results
        return self.results


class AbstractMlcFitterCreator(ABC):
    """
    The Creator class declares the factory method that is supposed to return an
    object of a Product class. The Creator's subclasses usually provide the
    implementation of this method.
    """

    @abstractmethod
    def mlc_fit_create(self, *args, **kwargs):
        """
        Note that the Creator may also provide some default implementation of
        the factory method.
        """
        pass

    def fit(self) -> dict:
        """
        Also note that, despite its name, the Creator's primary responsibility
        is not creating products. Usually, it contains some core business logic
        that relies on Product objects, returned by the factory method.
        Subclasses can indirectly change that business logic by overriding the
        factory method and returning a different type of product from it.
        """

        # Call the factory method to create a Product object.
        mlc_fitter = self.mlc_fit_create()

        # Now, use the product.
        result = mlc_fitter.fit()

        return result


class MlcDualAnnealingFitter(AbstractMlcFitter):
    def __init__(self, fit_dict:dict):
        self.init_theta = fit_dict['init_theta']
        self.LH_model = fit_dict['LH_model']
        self.maxiter = fit_dict['maxiter']
        self.results = dict()
        
    def fit(self) -> dict:     
        
        def fh(x):
            return -self.LH_model.get_logLH(x)

        print("init dual annealing...")
        res = dual_annealing(fh, self.init_theta, maxiter=self.maxiter)
        print("Dual annealing finished")
        self.results['mle_theta'] = res.x
        self.results['log_mle_value'] = -res.fun
        self.results['mle_simulated_data'] = self.LH_model.model.simulate_theta(self.LH_model.t,
                                            res.x)
        return self.results   

    
class bic_estimator:
    @staticmethod
    def bic(log_mle_value,k,N):    
        #k: no. of parameters
        #N: sample size
        BIC=k*np.log(N)-2*log_mle_value
        return BIC   

    
class MlcBasinHoppingFitter(AbstractMlcFitter):
    def __init__(self, fit_dict:dict):
        self.init_theta = fit_dict['init_theta']
        self.LH_model = fit_dict['LH_model']
        self.maxiter = fit_dict['maxiter']
        self.results = dict()
        
    def fit(self) -> dict:     
        
        def fh(x):
            return -self.LH_model.get_logLH(x)

        print("init basinhopping...")
        res = basinhopping(fh, self.init_theta, niter=self.maxiter)
        print("basinhopping finished")
        self.results['mle_theta'] = res.x
        self.results['log_mle_value'] = -res.fun
        self.results['mle_simulated_data'] = self.LH_model.model.simulate_theta(self.LH_model.t,
                                            res.x)
        return self.results
    
    
class MlcFitterCreator(AbstractMlcFitterCreator):
    """
    Note that the signature of the method still uses the abstract product type,
    even though the concrete product is actually returned from the method. This
    way the Creator can stay independent of concrete product classes.
    """

    def mlc_fit_create(self, fit_dict:dict):
        init_theta = fit_dict["init_theta"]
        init_list_filter = filter( lambda x: isinstance(x, (list, tuple)), init_theta)
        if len(list(init_list_filter)) == 0:
            #result_fitter = general_fit_dict['default_theta0_fitter'](fit_dict)
            result_fitter = MlcBasinHoppingFitter(fit_dict)
        else:
            #result_fitter = general_fit_dict['default_range_fitter'](fit_dict) 
            result_fitter = MlcDualAnnealingFitter(fit_dict)          
        return result_fitter