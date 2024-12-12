# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
from abc import ABC, abstractmethod

class AbstractLhEstimator(ABC):
    def __init__(self,t,data,model):
        self.t = t
        self.data = data
        self.model = model
        
    @abstractmethod
    def get_logLH(self,theta):
        pass
    
class AbstractLhModelCreator(ABC):
    """
    The Creator class declares the factory method that is supposed to return an
    object of a Product class. The Creator's subclasses usually provide the
    implementation of this method.
    """

    @abstractmethod
    def lh_model_create(self, *args, **kwargs):
        """
        Note that the Creator may also provide some default implementation of
        the factory method.
        """
        pass



class LhNormal(AbstractLhEstimator):
    def get_err_theta(self,theta):
        return theta[-1]
    
    def get_residual_correct_vector(self, theta):
        return 1

    def get_predictions(self,theta): 
        "before call model simulate theta in LhNormal"
        return self.model.simulate_theta(self.t.ravel(),theta)

    def get_logLH(self, theta):
        STD=self.get_err_theta(theta)        
        if np.isscalar(STD):
            LHs=[stats.norm(loc,STD).logpdf(self.data[i]) for i,loc in enumerate(self.get_predictions(theta))]
        else:
            LHs=[stats.norm(loc,STD[i]).logpdf(self.data[i]) for i,loc in enumerate(self.get_predictions(theta))]
        return np.sum(np.array(LHs))

class LhNormalProp(AbstractLhEstimator):
    def get_err_theta(self,theta):
        return theta[-1]
    
    def get_residual_correct_vector(self, theta):
        return self.data

    def get_predictions(self,theta):
        return self.model.simulate_theta(self.t.ravel(),theta)

    def get_logLH(self, theta):
        sigSTD=self.get_err_theta(theta)
        STD= sigSTD*self.data   
        if np.isscalar(STD):
            LHs=[stats.norm(loc,STD).logpdf(self.data[i]) for i,loc in enumerate(self.get_predictions(theta))]
        else:
            LHs=[stats.norm(loc,STD[i]).logpdf(self.data[i]) for i,loc in enumerate(self.get_predictions(theta))]
        return np.sum(np.array(LHs))
    
class LhBenz(AbstractLhEstimator):
    def get_err_theta(self,theta):
        return theta[-3:]
    
    def get_residual_correct_vector(self, theta):
        y = self.data
        E = np.zeros(len(y))
        alpha = theta[-1]
        Vm = theta[-2]
        for i,Y in enumerate(y):
            if Y<Vm:
                E[i]=Vm**alpha
            else:
                E[i]=Y**alpha
        return E

    def get_predictions(self,theta):
        return self.model.simulate_theta(self.t.ravel(),theta)

    def get_logLH(self, theta):
        sigma,Vm,alpha=self.get_err_theta(theta)

        E=np.zeros(len(self.data))
        for i,Y in enumerate(self.data):
            if Y<Vm:
                E[i]=Vm**alpha
            else:
                E[i]=Y**alpha

        STD=sigma*E
        LHs=[stats.norm(loc,STD[i]).logpdf(self.data[i]) for i,loc in enumerate(self.get_predictions(theta))]
        return np.sum(np.array(LHs))
    
class LhStudent(AbstractLhEstimator):
    def get_err_theta(self,theta):
        return [theta[-2],theta[-1]]
    
    def get_residual_correct_vector(self, theta):
        return 1

    def get_predictions(self,theta):
        return self.model.simulate_theta(self.t.ravel(),theta)

    def get_logLH(self, theta):
        [k,SCALE]=self.get_err_theta(theta)        
        if np.isscalar(SCALE):
            LHs=[stats.t.logpdf(self.data[i],k,loc,SCALE) for i,loc in enumerate(self.get_predictions(theta))]
        else:
            LHs=[stats.t.logpdf(self.data[i],k,loc,SCALE[i]) for i,loc in enumerate(self.get_predictions(theta))]
        return np.sum(np.array(LHs))
    
class LhStudentProp(AbstractLhEstimator):
    def get_err_theta(self,theta):
        return [theta[-2],theta[-1]]
    
    def get_residual_correct_vector(self, theta):
        return self.data

    def get_predictions(self,theta):
        return self.model.simulate_theta(self.t.ravel(),theta)

    def get_logLH(self, theta):
        [k,sigSCALE]=self.get_err_theta(theta) 
        SCALE= sigSCALE*self.data        
        if np.isscalar(SCALE):
            LHs=[stats.t.logpdf(self.data[i],k,loc,SCALE) for i,loc in enumerate(self.get_predictions(theta))]
        else:
            LHs=[stats.t.logpdf(self.data[i],k,loc,SCALE[i]) for i,loc in enumerate(self.get_predictions(theta))]
        return np.sum(np.array(LHs))
    
    
class LhModelCreator(AbstractLhModelCreator):
    """

    """

    def lh_model_create(self, model_dict:dict):
        lh_estimator_class = model_dict['lh_estimator_class']
        model = model_dict['model']
        data = model_dict['data']
        t = model_dict['t']
        lh_estimator = lh_estimator_class(t,data,model)
        return lh_estimator