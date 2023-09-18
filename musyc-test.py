from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm

from scipy.optimize import curve_fit
from scipy.stats import norm
import numpy as np

from scipy.stats import linregress

import inspect
import warnings
import pandas as pd

class ParametricModel(ABC):
    """Base class for paramterized synergy models, including MuSyC, Zimmer, GPDI, and BRAID.
    """
    def __init__(self):
        """Bounds for drug response parameters (for instance, given percent viability data, one might expect E to be bounded within (0,1)) can be set, or parameters can be explicitly set.
        """

        self.bounds = None
        self.fit_function = None
        self.jacobian_function = None
        
        self.converged = False

        self.sum_of_squares_residuals = None
        self.r_squared = None
        self.aic = None
        self.bic = None
        self.bootstrap_parameters = None

    def _score(self, d1, d2, E):
        """Calculate goodness of fit and model quality scores, including sum-of-squares residuals, R^2, Akaike Information Criterion (AIC), and Bayesian Information Criterion (BIC).

        If model is not yet paramterized, does nothing

        Called automatically during model.fit(d1, d2, E)

        Parameters
        ----------
        d1 : array_like
            Doses of drug 1
        
        d2 : array_like
            Doses of drug 2
        
        E : array_like
            Dose-response at doses d1 and d2
        """
        if (self._is_parameterized()):

            n_parameters = len(self._get_parameters())

            self.sum_of_squares_residuals = residual_ss(d1, d2, E, self.E)
            self.r_squared = r_squared(E, self.sum_of_squares_residuals)
            self.aic = AIC(self.sum_of_squares_residuals, n_parameters, len(E))
            self.bic = BIC(self.sum_of_squares_residuals, n_parameters, len(E))

    @abstractmethod
    def _get_parameters(self):
        """Returns all of the model's fit parameters

        Returns
        ----------
        parameters : list or tuple
            Model's parameters
        """
        pass

    @abstractmethod
    def get_parameters(self, confidence_interval=95):
        """Returns a dict of the model's parameters.

        When relevant, it will also return meaningful derived parameters. For instance, MuSyC has several parameters for E, but defines a synergy parameter beta as a function of E parameters. Thus, beta will also be included.
        
        If the model was fit to data with bootstrap_iterations > 0, this will also return the specified confidence interval.
        """
        pass

    @abstractmethod
    def summary(self, confidence_interval=95, tol=0.01):
        """Summarizes the model's synergy conclusions.

        For each synergy parameters, determines whether it indicates synergy or antagonism. When the model has been fit with bootstrap_parameters>0, the best fit, lower bound, and upper bound must all agree on synergy or antagonism.
    
        Parameters
        ----------
        confidence_interval : float, optional (default=95)
            If the model was fit() with bootstrap_parameters>0, confidence_interval will be used to get the upper and lower bounds. 

        tol : float, optional (default=0.01)
            Tolerance to determine synergy or antagonism. The parameter must exceed the threshold by at least tol (some parameters, like MuSyC's alpha which is antagonistic from 0 to 1, and synergistic from 1 to inf, will be log-scaled prior to comparison with tol)

        Returns
        ----------
        summary : str
            Tab-separated string. If the model has been bootstrapped, columns are [parameter, value, (lower,upper), synergism/antagonism]. If the model has not been bootstrapped, columns are [parameter, value, synergism/antagonism].
        """
        pass

    @abstractmethod
    def _get_single_drug_classes(self):
        """
        Returns
        -------
        default_single_class : class
            The default class type to use for single-drug models

        expected_single_superclass : class
            The required type for single-drug models. If a single-drug model is passed that is not an instance of this superclass, it will be re-instantiated using default_model
        """
        pass

    def _internal_fit(self, d, E, use_jacobian, verbose=True, **kwargs):
        """Internal method to fit the model to data (d,E)
        """
        try:
            if use_jacobian and self.jacobian_function is not None:
                popt, pcov = curve_fit(self.fit_function, d, E, bounds=self.bounds, jac=self.jacobian_function, **kwargs)
            else: 
                popt, pcov = curve_fit(self.fit_function, d, E, bounds=self.bounds, **kwargs)
            if True in np.isnan(popt):
                return None
            return self._transform_params_from_fit(popt)
        except Exception as err:
            if verbose:
                print("Exception during combination drug response fit: %s"%err)
            return None

    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None, use_jacobian = True, p0=None, bootstrap_iterations=0, seed=None, **kwargs):
        """Fit the model to data.

        Parameters
        ----------
        d1 : array_like
            Doses of drug 1
        
        d2 : array_like
            Doses of drug 2

        E : array_like
            Dose-response at doses d1 and d2

        drug1_model : single-drug-model, default=None
            Only used when p0 is None. Pre-defined, or fit, model (e.g., Hill()) of drug 1 alone. Parameters from this model are used to provide an initial guess of E0, E1, h1, and C1 for the 2D-model fit. If None (and p0 is None), then d1 and E will be masked where d2==min(d2), and used to fit a model for drug 1.

        drug2_model : single-drug-model, default=None
            Same as drug1_model, for drug 2.
        
        use_jacobian : bool, default=True
            If True, will use the Jacobian to help guide fit (ONLY MuSyC, Hill, and Hill_2P have Jacobian implemented yet). When the number
            of data points is less than a few hundred, this makes the fitting
            slower. However, it also improves the reliability with which a fit
            can be found. If drug1_model or drug2_model are None, use_jacobian will also be applied for their fits.

        p0 : tuple, default=None
            Initial guess for the parameters. If p0 is None (default), drug1_model and drug2_model will be used to obtain an initial guess. If they are also None, they will be fit to the data. If they fail to fit, the initial guess will be E0=max(E), Emax=min(E), h=1, C=median(d), and all synergy parameters are additive (i.e., at the boundary between antagonistic and synergistic)

        seed : int, default=None
            If not None, used as numpy.random.seed(start_seed) at the beginning of bootstrap resampling
        
        kwargs
            kwargs to pass to scipy.optimize.curve_fit()
        """

        if seed is not None: np.random.seed(seed)
        d1 = np.asarray(d1, dtype=np.float64)
        d2 = np.asarray(d2, dtype=np.float64)

        E = np.asarray(E)

        xdata = np.vstack((d1,d2))
        
        if 'p0' in kwargs:
            p0 = list(kwargs.get('p0'))
        else:
            p0 = None
        
        p0 = self._get_initial_guess(d1, d2, E, drug1_model, drug2_model, p0=p0)

        kwargs['p0']=p0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            popt = self._internal_fit(xdata, E, use_jacobian, **kwargs)

        if popt is None:
            self._set_parameters(self._transform_params_from_fit(p0))
            self.converged = False
        else:
            self.converged = True
            self._set_parameters(popt)
            n_parameters = len(popt)
            n_samples = len(d1)
            if (n_samples - n_parameters - 1 > 0):
                self._score(d1, d2, E)
                kwargs['p0'] = self._transform_params_to_fit(popt)
                self._bootstrap_resample(d1, d2, E, use_jacobian, bootstrap_iterations, **kwargs)
    
    @abstractmethod
    def E(self, d1, d2):
        """Returns drug effect E at dose d1,d2 for a pre-defined or fitted model.

        Parameters
        ----------
        d1 : array_like
            Doses of drug 1
        
        d2 : array_like
            Doses of drug 2
        
        Returns
        ----------
        effect : array_like
            Evaluate's the model at doses d1 and d2
        """
        pass

    def _is_parameterized(self):
        """Returns False if any parameters are None or nan.

        Returns
        ----------
        is_parameterzed : bool
            True if all of the parameters are set. False if any are None or nan.
        """
        return not (None in self._get_parameters() or True in np.isnan(np.asarray(self._get_parameters())))

    @abstractmethod
    def _set_parameters(self, popt):
        """Internal method to set model parameters
        """
        pass

    @abstractmethod
    def _transform_params_from_fit(self, params):
        """Internal method to transform parameterss as needed.

        For instance, models that fit logh and logC must transform those to h and C
        """
        pass

    @abstractmethod
    def _transform_params_to_fit(self, params):
        """Internal method to transform parameterss as needed.

        For instance, models that fit logh and logC must transform from h and C
        """
        pass

    @abstractmethod
    def _get_initial_guess(self, d1, d2, E, drug1_model, drug2_model, p0=None):
        """Internal method to format and/or guess p0
        """
        pass

    
    def _bootstrap_resample(self, d1, d2, E, use_jacobian, bootstrap_iterations, seed=None, **kwargs):
        """Internal function to identify confidence intervals for parameters
        """

        if not self._is_parameterized(): return
        if not self.converged: return

        n_data_points = len(E)
        n_parameters = len(self._get_parameters())
        
        sigma_residuals = np.sqrt(self.sum_of_squares_residuals / (n_data_points - n_parameters))

        E_model = self.E(d1, d2)
        bootstrap_parameters = []

        xdata = np.vstack((d1,d2))

        #if start_seed is not None: np.random.seed(start_seed)
        for iteration in range(bootstrap_iterations):
            #if start_seed is not None: np.random.seed(start_seed + iteration)
            residuals_step = norm.rvs(loc=0, scale=sigma_residuals, size=n_data_points)

            # Add random noise to model prediction
            E_iteration = E_model + residuals_step

            # Fit noisy data
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                popt1 = self._internal_fit(xdata, E_iteration, verbose=False, use_jacobian=use_jacobian, **kwargs)
            
            if popt1 is not None:
                bootstrap_parameters.append(popt1)
        if len(bootstrap_parameters) > 0:
            self.bootstrap_parameters = np.vstack(bootstrap_parameters)
        else:
            self.bootstrap_parameters = None

    def get_parameter_range(self, confidence_interval=95):
        """Returns the lower bound and upper bound estimate for each parameter.

        Parameters:
        -----------
        confidence_interval : int, float, default=95
            % confidence interval to return. Must be between 0 and 100.
        """
        if not self._is_parameterized():
            return None
        if not self.converged:
            return None
        if confidence_interval < 0 or confidence_interval > 100:
            return None
        if self.bootstrap_parameters is None:
            return None

        lb = (100-confidence_interval)/2.
        ub = 100-lb
        return np.percentile(self.bootstrap_parameters, [lb, ub], axis=0)

    def plot_heatmap(self, d1, d2, cmap="YlGnBu", **kwargs):
        """Plots the model's effect, E(d1, d2) as a heatmap

        Parameters
        ----------
        d1 : array_like
            Doses of drug 1
        
        d2 : array_like
            Doses of drug 2
        
        kwargs
            kwargs passed to synergy.plots.plot_heatmap()
        """
        if not self._is_parameterized():
            #raise ModelNotParameterizedError()
            return
        
        E = self.E(d1, d2)
        plot_heatmap(d1, d2, E, cmap=cmap, **kwargs)

    def plot_residual_heatmap(self, d1, d2, E, cmap="RdBu", center_on_zero=True, **kwargs):
        """Plots the residuals of the fit model as a heatmap

        Parameters
        ----------
        d1 : array_like
            Doses of drug 1
        
        d2 : array_like
            Doses of drug 2

        E : array_like
            Observed drug effects

        cmap : string, default="RdBu"
            Colormap for the plot
        
        kwargs
            kwargs passed to synergy.plots.plot_heatmap()
        """
        if not self._is_parameterized():
            #raise ModelNotParameterizedError()
            return
        
        Emodel = self.E(d1, d2)
        plot_heatmap(d1, d2, E-Emodel, cmap=cmap, center_on_zero=center_on_zero, **kwargs)

    @abstractmethod
    def _reference_E(self, d1, d2):
        pass

    def plot_reference_heatmap(self, d1, d2, cmap="YlGnBu", **kwargs):
        if not self._is_parameterized():
            #raise ModelNotParameterizedError()
            return

        Ereference = self._reference_E(d1, d2)
        plot_heatmap(d1, d2, Ereference, cmap=cmap, **kwargs)

    def plot_reference_surface(self, d1, d2, cmap="YlGnBu", **kwargs):
        if not self._is_parameterized():
            return
        Ereference = self._reference_E(d1, d2)
        plot_surface_plotly(d1, d2, Ereference, cmap=cmap, **kwargs)

    def plot_delta_heatmap(self, d1, d2, cmap="PRGn", center_on_zero=True, **kwargs):
        if not self._is_parameterized():
            #raise ModelNotParameterizedError()
            return
        Ereference = self._reference_E(d1, d2)
        Emodel = self.E(d1, d2)
        plot_heatmap(d1, d2, Ereference-Emodel, cmap=cmap, center_on_zero=center_on_zero, **kwargs)

    def plot_delta_surface(self, d1, d2, cmap="PRGn", center_on_zero=True, **kwargs):
        if not self._is_parameterized():
            return
        Ereference = self._reference_E(d1, d2)
        Emodel = self.E(d1, d2)
        plot_surface_plotly(d1, d2, Ereference-Emodel, cmap=cmap, center_on_zero=center_on_zero, **kwargs)
    

    def plot_surface_plotly(self, d1, d2, cmap="YlGnBu", **kwargs):
        """Plots the model's effect, E(d1, d2) as a surface using synergy.plots.plot_surface_plotly()

        Parameters
        ----------
        d1 : array_like
            Doses of drug 1
        
        d2 : array_like
            Doses of drug 2
        
        cmap : string, default="viridis"
            Colorscale for the plot

        kwargs
            kwargs passed to synergy.plots.plot_heatmap()
        """
        if not self._is_parameterized():
            #raise ModelNotParameterizedError()
            return
        
        # d1 and d2 may come from data, and have replicates. This would cause problems with surface plots (replicates in scatter_points are fine, but replicates in the surface itself are not)
        #d1, d2 = dose_tools.remove_replicates(d1, d2)
        E = self.E(d1, d2)
        plot_surface_plotly(d1, d2, E, cmap=cmap, **kwargs)



class ParameterizedModel1D:
    def __init__(self):
        self.bounds = None
        self.fit_function = None
        self.jacobian_function = None
        
        self.converged = False
        self._fit = False

        self.sum_of_squares_residuals = None
        self.r_squared = None
        self.aic = None
        self.bic = None
        self.bootstrap_parameters = None
    
    def _internal_fit(self, d, E, use_jacobian, verbose=True, **kwargs):
        """Internal method to fit the model to data (d,E)
        """
        try:
        #if True:
            if use_jacobian:
                popt, pcov = curve_fit(self.fit_function, d, E, bounds=self.bounds, jac=self.jacobian_function, **kwargs)
            else: 
                popt, pcov = curve_fit(self.fit_function, d, E, bounds=self.bounds, **kwargs)
            if True in np.isnan(popt):
                return None
            return self._transform_params_from_fit(popt)
        except Exception as err:
            if verbose:
                print("Exception during single drug response fit: %s"%err)
            return None

    def _get_initial_guess(self, d, E, p0=None):
        """Internal method to format and/or guess p0
        """
        return p0

    def _set_parameters(self, popt):
        """Internal method to set model parameters
        """
        pass

    def fit(self, d, E, use_jacobian=True, bootstrap_iterations=0, bootstrap_confidence_interval=95, **kwargs):
        """Fit the Hill equation to data. Fitting algorithm searches for h and C in a log-scale, but all bounds and guesses should be provided in a linear scale.

        Parameters
        ----------
        d : array_like
            Array of doses measured
        
        E : array_like
            Array of effects measured at doses d
        
        use_jacobian : bool, default=True
            If True, will use the Jacobian to help guide fit. When the number
            of data points is less than a few hundred, this makes the fitting
            slower. However, it also improves the reliability with which a fit
            can be found.
        
        kwargs
            kwargs to pass to scipy.optimize.curve_fit()
        """

        self._fit = True
        d = np.asarray(d)
        E = np.asarray(E)

        if 'p0' in kwargs:
            p0 = list(kwargs.get('p0'))
        else:
            p0 = None
        p0 = self._get_initial_guess(d, E, p0=p0)
        kwargs['p0']=p0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            popt = self._internal_fit(d, E, use_jacobian, **kwargs)

        if popt is None:
            self.converged = False
            self._set_parameters(self._transform_params_from_fit(p0))
        else:
            self.converged = True
            self._set_parameters(popt)

            n_parameters = len(popt)
            n_samples = len(d)
            if (n_samples - n_parameters - 1 > 0):
                self._score(d, E)
                kwargs['p0'] = self._transform_params_to_fit(popt)
                self._bootstrap_resample(d, E, use_jacobian, bootstrap_iterations, bootstrap_confidence_interval, **kwargs)

    def E(self, d):
        """Evaluate this model at dose d.

        Parameters
        ----------
        d : array_like
            Doses to calculate effect at
        
        Returns
        ----------
        effect : array_like
            Evaluate's the model at dose in d
        """
        ret = 0*d
        ret[:] = np.nan
        return ret
        
    def E_inv(self, E):
        """Evaluate the inverse of this model.

        Parameters
        ----------
        E : array_like
            Effects to get the doses for
        
        Returns
        ----------
        doses : array_like
            Doses which achieve effects E using this model. Will return np.nan for effects outside of the model's effect range, or for non-invertable models
        """
        ret = 0*d
        ret[:] = np.nan
        return ret

    def _transform_params_from_fit(self, params):
        """Internal method to transform parameterss as needed.

        For instance, models that fit logh and logC must transform those to h and C
        """
        return params

    def _transform_params_to_fit(self, params):
        """Internal method to transform parameterss as needed.

        For instance, models that fit logh and logC must transform from h and C
        """
        return params

    def get_parameters(self):
        """Returns model parameters
        """
        return []

    def get_parameter_range(self, confidence_interval=95):
        """Returns the lower bound and upper bound estimate for each parameter.

        Parameters:
        -----------
        confidence_interval : int, float, default=95
            % confidence interval to return. Must be between 0 and 100.
        """
        if not self._is_parameterized():
            return None
        if not self.converged:
            return None
        if confidence_interval < 0 or confidence_interval > 100:
            return None
        if self.bootstrap_parameters is None:
            return None

        lb = (100-confidence_interval)/2.
        ub = 100-lb
        return np.percentile(self.bootstrap_parameters, [lb, ub], axis=0)

    def _is_parameterized(self):
        """Internalized method to check all model parameters are set
        """
        return not (None in self.get_parameters() or True in np.isnan(np.asarray(self.get_parameters())))

    def _score(self, d, E):
        """Calculate goodness of fit and model quality scores, including sum-of-squares residuals, R^2, Akaike Information Criterion (AIC), and Bayesian Information Criterion (BIC).

        If model is not yet paramterized, does nothing

        Called automatically during model.fit(d1, d2, E)

        Parameters
        ----------
        d : array_like
            Doses
        
        E : array_like
            Measured dose-response at doses d
        """
        if (self._is_parameterized()):

            n_parameters = len(self.get_parameters())

            self.sum_of_squares_residuals = residual_ss_1d(d, E, self.E)
            self.r_squared = r_squared(E, self.sum_of_squares_residuals)
            self.aic = AIC(self.sum_of_squares_residuals, n_parameters, len(E))
            self.bic = BIC(self.sum_of_squares_residuals, n_parameters, len(E))

    def _bootstrap_resample(self, d, E, use_jacobian, bootstrap_iterations, confidence_interval, **kwargs):
        """Internal function to identify confidence intervals for parameters
        """

        if not self._is_parameterized(): return
        if not self.converged: return

        n_data_points = len(E)
        n_parameters = len(self.get_parameters())
        
        sigma_residuals = np.sqrt(self.sum_of_squares_residuals / (n_data_points - n_parameters))

        E_model = self.E(d)
        bootstrap_parameters = []

        for iteration in range(bootstrap_iterations):
            residuals_step = norm.rvs(loc=0, scale=sigma_residuals, size=n_data_points)

            # Add random noise to model prediction
            E_iteration = E_model + residuals_step

            # Fit noisy data
            with np.errstate(divide='ignore', invalid='ignore'):
                popt1 = self._internal_fit(d, E_iteration, verbose=False, use_jacobian=use_jacobian, **kwargs)
            
            if popt1 is not None:
                bootstrap_parameters.append(popt1)
        
        if len(bootstrap_parameters) > 0:
            self.bootstrap_parameters = np.vstack(bootstrap_parameters)
        else:
            self.bootstrap_parameters = None

    def is_fit(self):
        return self._is_parameterized()
        #return self._fit

def jacobian(d1, d2, E0, E1, E2, E3, logh1, logh2, logC1, logC2, r1r, r2r, logalpha12, logalpha21, loggamma12, loggamma21):
    """Evaluates Jacobian of MuSyC (gamma)
    
    Returns:
    -----------
    jacobian : tuple
        j_E0, j_E1, j_E2, j_E3, j_logh1, j_logh2, j_logC1, j_logC2, j_logalpha12, j_logalpha21, j_loggamma12, j_loggamma21
    """
    h1 = np.exp(logh1)
    h2 = np.exp(logh2)
    C1 = np.exp(logC1)
    C2 = np.exp(logC2)
    alpha12 = np.exp(logalpha12)
    alpha21 = np.exp(logalpha21)
    gamma12 = np.exp(loggamma12)
    gamma21 = np.exp(loggamma21)

    logd1 = np.log(d1)
    logd2 = np.log(d2)

    logd1alpha21 = np.log(d1*alpha21)
    logd2alpha12 = np.log(d2*alpha12)

    #d1h1 = d1**h1
    #d2h2 = d2**h2
    d1h1 = np.power(d1,h1)
    d2h2 = np.power(d2,h2)

    #C1h1 = C1**h1
    #C2h2 = C2**h2
    C1h1 = np.power(C1,h1)
    C2h2 = np.power(C2,h2)

    r1 = r1r/C1h1
    r2 = r2r/C2h2

    #alpha21d1gamma21h1 = (alpha21*d1)**(gamma21*h1)
    #alpha12d2gamma12h2 = (alpha12*d2)**(gamma12*h2)
    alpha21d1gamma21h1 = np.power(alpha21*d1, gamma21*h1)
    alpha12d2gamma12h2 = np.power(alpha12*d2, gamma12*h2)

    #C12h1 = C1**(2*h1)
    #C22h2 = C2**(2*h2)
    C12h1 = np.power(C1,2*h1)
    C22h2 = np.power(C2,2*h2)

    exp = np.exp
    log = np.log

    # ********** logh1 ********

    j_logh1 = E0*(r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)*(-d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*gamma21*h1*logC1 - d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*h1*logd1 - d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*h1*logd1 - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*h1*logd1 - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*h1*logC1 - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*h1*logd1 - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*h1*logd1 - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1)*logd1alpha21 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*h1*logd1 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1)*logd1alpha21 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*gamma21*h1*logC1 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*h1*logC1 - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*h1*logC1 - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*h1*logC1 - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*(gamma21*h1)*logd1alpha21 - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1)*logd1alpha21 - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1)*logd1alpha21 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*h1*logC1 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*gamma21*h1*logC1 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*h1*logC1 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*h1*logC1 - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*h1*logC1 - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*(gamma21*h1)*logd1alpha21 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*gamma21*h1*logC1)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E0*(r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*gamma21*h1*logC1 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*h1*logC1 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*h1*logC1 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*h1*logC1 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*(gamma21*h1)*logd1alpha21 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*gamma21*h1*logC1)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E1*(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12)*(-d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*gamma21*h1*logC1 - d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*h1*logd1 - d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*h1*logd1 - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*h1*logd1 - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*h1*logC1 - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*h1*logd1 - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*h1*logd1 - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1)*logd1alpha21 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*h1*logd1 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1)*logd1alpha21 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*gamma21*h1*logC1 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*h1*logC1 - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*h1*logC1 - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*h1*logC1 - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*(gamma21*h1)*logd1alpha21 - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1)*logd1alpha21 - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1)*logd1alpha21 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*h1*logC1 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*gamma21*h1*logC1 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*h1*logC1 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*h1*logC1 - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*h1*logC1 - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*(gamma21*h1)*logd1alpha21 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*gamma21*h1*logC1)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E1*(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*gamma21*h1*logC1 + d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*h1*logd1 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*h1*logd1 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*h1*logd1 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1)*logd1alpha21 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1)*logd1alpha21)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E2*(d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21)*(-d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*gamma21*h1*logC1 - d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*h1*logd1 - d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*h1*logd1 - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*h1*logd1 - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*h1*logC1 - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*h1*logd1 - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*h1*logd1 - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1)*logd1alpha21 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*h1*logd1 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1)*logd1alpha21 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*gamma21*h1*logC1 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*h1*logC1 - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*h1*logC1 - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*h1*logC1 - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*(gamma21*h1)*logd1alpha21 - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1)*logd1alpha21 - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1)*logd1alpha21 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*h1*logC1 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*gamma21*h1*logC1 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*h1*logC1 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*h1*logC1 - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*h1*logC1 - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*(gamma21*h1)*logd1alpha21 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*gamma21*h1*logC1)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E2*(d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*h1*logC1 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*h1*logd1 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*gamma21*h1*logC1 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*h1*logC1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*h1*logC1 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*h1*logC1)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E3*(d2h2*r1*r2*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*C1h1*h1*logC1 + d2h2*r2*(r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d1h1*r1*(d1h1*r1*h1*logd1 + r1*C1h1*h1*logC1) + d1h1*r1*(-d1h1*r1 - r1*C1h1 - r2**gamma12*alpha12d2gamma12h2)*h1*logd1 + d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1)*h1*logd1 + (-d1h1*r1*h1*logd1 - r1*C1h1*h1*logC1)*(d1h1*r1 + d2h2*r2 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2*h1*logd1 + r1**gamma21*alpha21d1gamma21h1*(-d1h1*r1 - r1*C1h1 - r2**gamma12*alpha12d2gamma12h2)*(gamma21*h1)*logd1alpha21 + (-d1h1*r1*h1*logd1 - r1*C1h1*h1*logC1)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2))*(d1h1*r1*(d1h1*r1*h1*logd1 + r1*C1h1*h1*logC1) + d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1)*h1*logd1 - d1h1*r1*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*h1*logd1 + (-d1h1*r1 - d2h2*r2)*(d1h1*r1*h1*logd1 + r1*C1h1*h1*logC1)))/(-((-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) + (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))) + E3*(d2h2*r2*(r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)))*(-(-(-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2*h1*logd1 + r1**gamma21*alpha21d1gamma21h1*(-d1h1*r1 - r1*C1h1 - r2**gamma12*alpha12d2gamma12h2)*(gamma21*h1)*logd1alpha21 + (-d1h1*r1*h1*logd1 - r1*C1h1*h1*logC1)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) - (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d1h1*d2h2*r1*r2*h1*logd1 - (r1*C1h1)**gamma21*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*gamma21*h1*logC1 + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1*h1*logd1 + r1*C1h1*h1*logC1)) - (-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*r1*(d1h1*r1*h1*logd1 + r1*C1h1*h1*logC1) + d1h1*r1*(-d1h1*r1 - r1*C1h1 - r2**gamma12*alpha12d2gamma12h2)*h1*logd1 + d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1)*h1*logd1 + (-d1h1*r1*h1*logd1 - r1*C1h1*h1*logC1)*(d1h1*r1 + d2h2*r2 + r2*C2h2)) - (d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2))*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1)*h1*logd1 - d1h1*r1*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*h1*logd1 - (-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1*h1*logd1 + r1*C1h1*h1*logC1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1*h1*logd1 + r1*C1h1*h1*logC1)))/(-((-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) + (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)))**2

    # ********** logh2 ********

    j_logh2 = E0*(r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)*(-d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*h2*logC2 - d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*gamma12*h2*logC2 - d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*h2*logC2 - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*h2*logC2 - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*(gamma12*h2)*logd2alpha12 - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2)*logd2alpha12 - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*h2*logC2 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*h2*logd2 - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*gamma12*h2*logC2 - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*h2*logd2 - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*h2*logd2 - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*h2*logC2 - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*h2*logd2 - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*h2*logd2 - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*h2*logd2 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2)*logd2alpha12 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*h2*logC2 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*gamma12*h2*logC2 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*h2*logC2 - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*gamma12*h2*logC2 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*h2*logC2 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*(gamma12*h2)*logd2alpha12)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E0*(r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*h2*logC2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*gamma12*h2*logC2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*h2*logC2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*gamma12*h2*logC2 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*h2*logC2 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*(gamma12*h2)*logd2alpha12)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E1*(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12)*(-d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*h2*logC2 - d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*gamma12*h2*logC2 - d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*h2*logC2 - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*h2*logC2 - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*(gamma12*h2)*logd2alpha12 - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2)*logd2alpha12 - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*h2*logC2 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*h2*logd2 - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*gamma12*h2*logC2 - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*h2*logd2 - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*h2*logd2 - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*h2*logC2 - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*h2*logd2 - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*h2*logd2 - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*h2*logd2 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2)*logd2alpha12 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*h2*logC2 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*gamma12*h2*logC2 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*h2*logC2 - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*gamma12*h2*logC2 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*h2*logC2 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*(gamma12*h2)*logd2alpha12)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E1*(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*h2*logC2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*gamma12*h2*logC2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*h2*logC2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*h2*logC2 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*h2*logC2 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*h2*logd2)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E2*(d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21)*(-d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*h2*logC2 - d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*gamma12*h2*logC2 - d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*h2*logC2 - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*h2*logC2 - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*(gamma12*h2)*logd2alpha12 - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2)*logd2alpha12 - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*h2*logC2 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*h2*logd2 - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*gamma12*h2*logC2 - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*h2*logd2 - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*h2*logd2 - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*h2*logC2 - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*h2*logd2 - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*h2*logd2 - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*h2*logd2 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2)*logd2alpha12 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*h2*logC2 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*gamma12*h2*logC2 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*h2*logC2 - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*gamma12*h2*logC2 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*h2*logC2 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*(gamma12*h2)*logd2alpha12)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E2*(d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2)*logd2alpha12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*h2*logd2 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*gamma12*h2*logC2 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*h2*logd2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*h2*logd2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2)*logd2alpha12)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E3*(d2h2*r2*r2**gamma12*alpha12d2gamma12h2*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(gamma12*h2)*logd2alpha12 + d2h2*r2*(r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*h2*logd2 + d2h2*r2*(r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d1h1*d2h2*r1*r2*h2*logd2 - r2**gamma12*alpha12d2gamma12h2*(d1h1*r1 + d2h2*r2 + r2*C2h2)*(gamma12*h2)*logd2alpha12 + (d2h2*r2*h2*logd2 + r2*C2h2*h2*logC2)*(-d1h1*r1 - r1*C1h1 - r2**gamma12*alpha12d2gamma12h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2*h2*logd2 - r2**gamma12*alpha12d2gamma12h2*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)*(gamma12*h2)*logd2alpha12 + (d2h2*r2*h2*logd2 + r2*C2h2*h2*logC2)*(-d1h1*r1 - r1*C1h1 - r2**gamma12*alpha12d2gamma12h2)) + (d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2))*(d1h1*d2h2*r1*r2*h2*logd2 - d2h2*r2*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*h2*logd2 + r2**gamma12*alpha12d2gamma12h2*(-d1h1*r1 - d2h2*r2)*(gamma12*h2)*logd2alpha12))/(-((-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) + (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))) + E3*(d2h2*r2*(r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)))*(-(-(-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2*h2*logd2 - r2**gamma12*alpha12d2gamma12h2*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)*(gamma12*h2)*logd2alpha12 + (d2h2*r2*h2*logd2 + r2*C2h2*h2*logC2)*(-d1h1*r1 - r1*C1h1 - r2**gamma12*alpha12d2gamma12h2)) - (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d2h2*r2*(r2*C2h2)**gamma12*gamma12*h2*logC2 - d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12)*h2*logd2 + d2h2*r2*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*h2*logd2 + r2**gamma12*alpha12d2gamma12h2*(d2h2*r2 - (r1*C1h1)**gamma21)*(gamma12*h2)*logd2alpha12) - (-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2*h2*logd2 - r2**gamma12*alpha12d2gamma12h2*(d1h1*r1 + d2h2*r2 + r2*C2h2)*(gamma12*h2)*logd2alpha12 + (d2h2*r2*h2*logd2 + r2*C2h2*h2*logC2)*(-d1h1*r1 - r1*C1h1 - r2**gamma12*alpha12d2gamma12h2)) - (d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2))*(-d2h2*r2*(-d1h1*r1 + (r2*C2h2)**gamma12)*h2*logd2 - d2h2*r2*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*h2*logd2 - r2**gamma12*alpha12d2gamma12h2*(d1h1*r1 + d2h2*r2)*(gamma12*h2)*logd2alpha12 - (r2*C2h2)**gamma12*(d1h1*r1 + d2h2*r2 + r1*C1h1)*gamma12*h2*logC2))/(-((-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) + (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)))**2

    # ********** logC1 ********

    j_logC1 = E0*(r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)*(-d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*gamma21*h1 - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*h1 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*gamma21*h1 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*h1 - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*h1 - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*h1 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*h1 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*gamma21*h1 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*h1 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*h1 - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*h1 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*gamma21*h1)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E0*(r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*gamma21*h1 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*h1 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*h1 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*gamma21*h1)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E1*d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*gamma21*h1/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E1*(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12)*(-d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*gamma21*h1 - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*h1 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*gamma21*h1 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*h1 - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*h1 - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*h1 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*h1 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*gamma21*h1 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*h1 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*h1 - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*h1 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*gamma21*h1)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E2*(d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21)*(-d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*gamma21*h1 - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*h1 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*gamma21*h1 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*h1 - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*h1 - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*h1 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*h1 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*gamma21*h1 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*h1 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*h1 - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*h1 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*gamma21*h1)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E2*(d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*h1 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*gamma21*h1 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*h1 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*h1)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E3*(d2h2*r1*r2*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*C1h1*h1 + d2h2*r2*(r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d1h1*r1**2*C1h1*h1 - r1*(d1h1*r1 + d2h2*r2 + r2*C2h2)*C1h1*h1) - r1*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)*C1h1*h1 + (d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2))*(d1h1*r1**2*C1h1*h1 + r1*(-d1h1*r1 - d2h2*r2)*C1h1*h1))/(-((-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) + (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))) + E3*(d2h2*r2*(r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)))*(r1*(-(-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)*C1h1*h1 - (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(r1*(d2h2*r2 - (r1*C1h1)**gamma21)*C1h1*h1 - (r1*C1h1)**gamma21*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*gamma21*h1) - (-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*r1**2*C1h1*h1 - r1*(d1h1*r1 + d2h2*r2 + r2*C2h2)*C1h1*h1) - (d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2))*(-r1*(-d1h1*r1 + (r2*C2h2)**gamma12)*C1h1*h1 - r1*(d1h1*r1 + d2h2*r2)*C1h1*h1))/(-((-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) + (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)))**2

    # ********** logC2 ********

    j_logC2 = E0*(r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)*(-d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*h2 - d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*gamma12*h2 - d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*h2 - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*h2 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*h2 - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*gamma12*h2 - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*h2 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*h2 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*gamma12*h2 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*h2 - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*gamma12*h2 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*h2)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E0*(r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*gamma12*h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*gamma12*h2 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*h2)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E1*(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12)*(-d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*h2 - d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*gamma12*h2 - d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*h2 - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*h2 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*h2 - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*gamma12*h2 - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*h2 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*h2 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*gamma12*h2 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*h2 - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*gamma12*h2 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*h2)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E1*(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*gamma12*h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*h2 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*h2)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E2*d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*gamma12*h2/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E2*(d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21)*(-d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*h2 - d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*gamma12*h2 - d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*h2 - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*h2 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*h2 - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*gamma12*h2 - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*h2 - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*h2 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*gamma12*h2 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*h2 - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*gamma12*h2 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*h2)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E3*(d2h2*r2**2*(r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(-d1h1*r1 - r1*C1h1 - r2**gamma12*alpha12d2gamma12h2)*C2h2*h2 + r2*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d1h1*r1 - r1*C1h1 - r2**gamma12*alpha12d2gamma12h2)*C2h2*h2)/(-((-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) + (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))) + E3*(d2h2*r2*(r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)))*(-d2h2*r2*(r2*C2h2)**gamma12*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*gamma12*h2 - r2*(-(-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d1h1*r1 - r1*C1h1 - r2**gamma12*alpha12d2gamma12h2)*C2h2*h2 - r2*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d1h1*r1 - r1*C1h1 - r2**gamma12*alpha12d2gamma12h2)*C2h2*h2 + (r2*C2h2)**gamma12*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2))*(d1h1*r1 + d2h2*r2 + r1*C1h1)*gamma12*h2)/(-((-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) + (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)))**2

    # ********** logalpha12 ********

    j_logalpha12 = E0*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*(gamma12*h2)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E0*(r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)*(-d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*(gamma12*h2) - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2) - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2) - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2) - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2) - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*(gamma12*h2))/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E1*(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12)*(-d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*(gamma12*h2) - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2) - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2) - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2) - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2) - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*(gamma12*h2))/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E2*(d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2) + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2))/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E2*(d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21)*(-d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*(gamma12*h2) - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2) - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2) - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2) - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2) - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*(gamma12*h2))/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E3*(-d2h2*r2*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d1h1*r1 + d2h2*r2 + r2*C2h2)*(gamma12*h2) + d2h2*r2*r2**gamma12*alpha12d2gamma12h2*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(gamma12*h2) + r2**gamma12*alpha12d2gamma12h2*(-d1h1*r1 - d2h2*r2)*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2))*(gamma12*h2) - r2**gamma12*alpha12d2gamma12h2*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)*(gamma12*h2))/(-((-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) + (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))) + E3*(d2h2*r2*(r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)))*(r2**gamma12*alpha12d2gamma12h2*(d1h1*r1 + d2h2*r2)*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2))*(gamma12*h2) - r2**gamma12*alpha12d2gamma12h2*(d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(gamma12*h2) + r2**gamma12*alpha12d2gamma12h2*(-(-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)*(gamma12*h2) + r2**gamma12*alpha12d2gamma12h2*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*r1 + d2h2*r2 + r2*C2h2)*(gamma12*h2))/(-((-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) + (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)))**2

    # ********** logalpha21 ********

    j_logalpha21 = E0*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*(gamma21*h1)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E0*(r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)*(-d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1) - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1) - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*(gamma21*h1) - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1) - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1) - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*(gamma21*h1))/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E1*(d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1) + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1))/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E1*(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12)*(-d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1) - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1) - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*(gamma21*h1) - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1) - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1) - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*(gamma21*h1))/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E2*(d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21)*(-d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1) - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1) - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*(gamma21*h1) - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1) - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1) - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*(gamma21*h1))/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 - E3*r1**gamma21*alpha21d1gamma21h1*(-(-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d2h2*r2*(r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)))*(-d1h1*r1 - r1*C1h1 - r2**gamma12*alpha12d2gamma12h2)*(gamma21*h1)/(-((-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) + (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)))**2 + E3*r1**gamma21*alpha21d1gamma21h1*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d1h1*r1 - r1*C1h1 - r2**gamma12*alpha12d2gamma12h2)*(gamma21*h1)/(-((-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) + (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)))

    # ********** loggamma12 ********

    j_loggamma12 = E0*(r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)*(-d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*gamma12*log(r2*C2h2) - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*gamma12*log(r2) - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*(gamma12*h2)*logd2alpha12 - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma12*log(r2) - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2)*logd2alpha12 - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*gamma12*log(r2) - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*log(r2*C2h2) - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*gamma12*log(r2*C2h2) - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*log(r2*C2h2) - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*gamma12*log(r2) - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma12*log(r2) - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2)*logd2alpha12 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*gamma12*log(r2*C2h2) - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*gamma12*log(r2*C2h2) - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*gamma12*log(r2) - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*(gamma12*h2)*logd2alpha12)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E0*(r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*gamma12*log(r2*C2h2) + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*gamma12*log(r2*C2h2) + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*gamma12*log(r2) + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*(gamma12*h2)*logd2alpha12)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E1*(d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*gamma12*log(r2*C2h2) + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*log(r2*C2h2) + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*log(r2*C2h2))/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E1*(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12)*(-d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*gamma12*log(r2*C2h2) - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*gamma12*log(r2) - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*(gamma12*h2)*logd2alpha12 - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma12*log(r2) - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2)*logd2alpha12 - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*gamma12*log(r2) - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*log(r2*C2h2) - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*gamma12*log(r2*C2h2) - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*log(r2*C2h2) - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*gamma12*log(r2) - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma12*log(r2) - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2)*logd2alpha12 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*gamma12*log(r2*C2h2) - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*gamma12*log(r2*C2h2) - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*gamma12*log(r2) - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*(gamma12*h2)*logd2alpha12)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E2*(d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21)*(-d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2*gamma12*log(r2*C2h2) - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*gamma12*log(r2) - d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2*(gamma12*h2)*logd2alpha12 - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma12*log(r2) - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2)*logd2alpha12 - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*gamma12*log(r2) - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*log(r2*C2h2) - d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*gamma12*log(r2*C2h2) - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma12*log(r2*C2h2) - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*gamma12*log(r2) - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma12*log(r2) - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2)*logd2alpha12 - r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2*gamma12*log(r2*C2h2) - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*gamma12*log(r2*C2h2) - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*gamma12*log(r2) - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*(gamma12*h2)*logd2alpha12)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E2*(d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma12*log(r2) + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2)*logd2alpha12 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1*gamma12*log(r2*C2h2) + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma12*log(r2) + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*(gamma12*h2)*logd2alpha12)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E3*(d2h2*r2*(r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(-r2**gamma12*alpha12d2gamma12h2*gamma12*log(r2) - r2**gamma12*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12)*(d1h1*r1 + d2h2*r2 + r2*C2h2) + d2h2*r2*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(r2**gamma12*alpha12d2gamma12h2*gamma12*log(r2) + r2**gamma12*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12) + (-d1h1*r1 - d2h2*r2)*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2))*(r2**gamma12*alpha12d2gamma12h2*gamma12*log(r2) + r2**gamma12*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-r2**gamma12*alpha12d2gamma12h2*gamma12*log(r2) - r2**gamma12*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2))/(-((-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) + (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))) + E3*(d2h2*r2*(r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)))*(-(-(-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-r2**gamma12*alpha12d2gamma12h2*gamma12*log(r2) - r2**gamma12*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2) - (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d2h2*r2*(r2*C2h2)**gamma12*gamma12*log(r2*C2h2) + (d2h2*r2 - (r1*C1h1)**gamma21)*(r2**gamma12*alpha12d2gamma12h2*gamma12*log(r2) + r2**gamma12*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12)) - (-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-r2**gamma12*alpha12d2gamma12h2*gamma12*log(r2) - r2**gamma12*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12)*(d1h1*r1 + d2h2*r2 + r2*C2h2) - (d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2))*(-(r2*C2h2)**gamma12*(d1h1*r1 + d2h2*r2 + r1*C1h1)*gamma12*log(r2*C2h2) - (d1h1*r1 + d2h2*r2)*(r2**gamma12*alpha12d2gamma12h2*gamma12*log(r2) + r2**gamma12*alpha12d2gamma12h2*(gamma12*h2)*logd2alpha12)))/(-((-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) + (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)))**2

    # ********** loggamma21 ********

    j_loggamma21 = E0*(r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)*(-d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*gamma21*log(r1*C1h1) - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*log(r1*C1h1) - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*gamma21*log(r1) - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1)*logd1alpha21 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma21*log(r1) - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1)*logd1alpha21 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*gamma21*log(r1*C1h1) - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*gamma21*log(r1) - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*(gamma21*h1)*logd1alpha21 - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma21*log(r1) - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1)*logd1alpha21 - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*gamma21*log(r1) - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1)*logd1alpha21 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*log(r1*C1h1) - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*gamma21*log(r1*C1h1) - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*gamma21*log(r1) - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*(gamma21*h1)*logd1alpha21 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*gamma21*log(r1*C1h1))/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E0*(r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*gamma21*log(r1*C1h1) + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*gamma21*log(r1) + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*(gamma21*h1)*logd1alpha21 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*gamma21*log(r1*C1h1))/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E1*(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12)*(-d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*gamma21*log(r1*C1h1) - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*log(r1*C1h1) - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*gamma21*log(r1) - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1)*logd1alpha21 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma21*log(r1) - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1)*logd1alpha21 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*gamma21*log(r1*C1h1) - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*gamma21*log(r1) - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*(gamma21*h1)*logd1alpha21 - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma21*log(r1) - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1)*logd1alpha21 - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*gamma21*log(r1) - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1)*logd1alpha21 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*log(r1*C1h1) - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*gamma21*log(r1*C1h1) - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*gamma21*log(r1) - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*(gamma21*h1)*logd1alpha21 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*gamma21*log(r1*C1h1))/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E1*(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*gamma21*log(r1*C1h1) + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma21*log(r1) + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1)*logd1alpha21 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma21*log(r1) + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1)*logd1alpha21)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E2*(d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*log(r1*C1h1) + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*gamma21*log(r1*C1h1) + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*log(r1*C1h1))/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2) + E2*(d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21)*(-d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2*gamma21*log(r1*C1h1) - d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*log(r1*C1h1) - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*gamma21*log(r1) - d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1)*logd1alpha21 - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma21*log(r1) - d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1)*logd1alpha21 - d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1*gamma21*log(r1*C1h1) - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*gamma21*log(r1) - d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1*(gamma21*h1)*logd1alpha21 - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*gamma21*log(r1) - d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12*(gamma21*h1)*logd1alpha21 - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*gamma21*log(r1) - d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2*(gamma21*h1)*logd1alpha21 - d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*gamma21*log(r1*C1h1) - r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2*gamma21*log(r1*C1h1) - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*gamma21*log(r1) - r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1*(gamma21*h1)*logd1alpha21 - r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2*gamma21*log(r1*C1h1))/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)**2 + E3*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(r1**gamma21*alpha21d1gamma21h1*gamma21*log(r1) + r1**gamma21*alpha21d1gamma21h1*(gamma21*h1)*logd1alpha21)*(-d1h1*r1 - r1*C1h1 - r2**gamma12*alpha12d2gamma12h2)/(-((-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) + (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))) + E3*(d2h2*r2*(r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)))*((r1*C1h1)**gamma21*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*gamma21*log(r1*C1h1) - (-(-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(r1**gamma21*alpha21d1gamma21h1*gamma21*log(r1) + r1**gamma21*alpha21d1gamma21h1*(gamma21*h1)*logd1alpha21)*(-d1h1*r1 - r1*C1h1 - r2**gamma12*alpha12d2gamma12h2))/(-((-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) + (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)))**2

    # ********** E0 ********

    j_E0 = (r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)

    # ********** E1 ********

    j_E1 = (d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)

    # ********** E2 ********

    j_E2 = (d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21)/(d1h1*r1*r2*(r1*C1h1)**gamma21*C2h2 + d1h1*r1*r2*(r2*C2h2)**gamma12*C2h2 + d1h1*r1*r2**(gamma12 + 1)*alpha12d2gamma12h2*C2h2 + d1h1*r1*r2**gamma12*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + d1h1*r1**(gamma21 + 1)*r2**gamma12*alpha21d1gamma21h1*alpha12d2gamma12h2 + d1h1*r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1*r2*(r1*C1h1)**gamma21*C1h1 + d2h2*r1*r2*(r2*C2h2)**gamma12*C1h1 + d2h2*r1**(gamma21 + 1)*r2*alpha21d1gamma21h1*C1h1 + d2h2*r1**gamma21*r2*alpha21d1gamma21h1*(r2*C2h2)**gamma12 + d2h2*r1**gamma21*r2**(gamma12 + 1)*alpha21d1gamma21h1*alpha12d2gamma12h2 + d2h2*r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21 + r1*r2*(r1*C1h1)**gamma21*C1h1*C2h2 + r1*r2*(r2*C2h2)**gamma12*C1h1*C2h2 + r1**(gamma21 + 1)*alpha21d1gamma21h1*(r2*C2h2)**gamma12*C1h1 + r2**(gamma12 + 1)*alpha12d2gamma12h2*(r1*C1h1)**gamma21*C2h2)

    # ********** E3 ********

    j_E3 = (d2h2*r2*(r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)))/(-((-d1h1*r1 + (r2*C2h2)**gamma12)*(d1h1*r1 + d2h2*r2 + r1*C1h1) + (d1h1*r1 + d2h2*r2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(d1h1*d2h2*r1*r2 - (d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)*(d2h2*r2 + r1**gamma21*alpha21d1gamma21h1 + r2*C2h2)) + (d1h1*r1*(d1h1*r1 + d2h2*r2 + r1*C1h1) - (d1h1*r1 + d2h2*r2 + r2*C2h2)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2))*(-d2h2*r2*(d1h1*r1 - (r2*C2h2)**gamma12) + (d2h2*r2 - (r1*C1h1)**gamma21)*(d1h1*r1 + r1*C1h1 + r2**gamma12*alpha12d2gamma12h2)))





    # E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21
    jac = np.hstack([j.reshape(-1,1) for j in [j_E0, j_E1, j_E2, j_E3, j_logh1, j_logh2, j_logC1, j_logC2, j_logalpha12, j_logalpha21, j_loggamma12, j_loggamma21]])
    jac[np.isnan(jac)] = 0
    return jac


class Hill(ParameterizedModel1D):
    """The four-parameter Hill equation

                            d^h
    E = E0 + (Emax-E0) * ---------
                         C^h + d^h

    The Hill equation is a standard model for single-drug dose-response curves.
    This is the base model for Hill_2P and Hill_CI.

    """
    def __init__(self, E0=None, Emax=None, h=None, C=None, E0_bounds=(-np.inf, np.inf), Emax_bounds=(-np.inf, np.inf), h_bounds=(0,np.inf), C_bounds=(0,np.inf)):
        """
        Parameters
        ----------
        E0 : float, optional
            Effect at 0 dose. Set this if you are creating a synthetic Hill
            model, rather than fitting from data
        
        Emax : float, optional
            Effect at 0 dose. Set this if you are creating a synthetic Hill
            model, rather than fitting from data
        
        h : float, optional
            The Hill-slope. Set this if you are creating a synthetic Hill
            model, rather than fitting from data
        
        C : float, optional
            EC50, the dose for which E = (E0+Emax)/2. Set this if you are
            creating a synthetic Hill model, rather than fitting from data
        
        X_bounds: tuple
            Bounds to use for Hill equation parameters during fitting. Valid options are E0_bounds, Emax_bounds, h_bounds, C_bounds.
        """

        super().__init__()
        self.E0 = E0
        self.Emax = Emax
        self.h = h
        self.C = C
        
        self.E0_bounds=E0_bounds
        self.Emax_bounds=Emax_bounds
        self.h_bounds=h_bounds
        self.C_bounds=C_bounds
        with np.errstate(divide='ignore'):
            self.logh_bounds = (np.log(h_bounds[0]), np.log(h_bounds[1]))
            self.logC_bounds = (np.log(C_bounds[0]), np.log(C_bounds[1]))

        self.fit_function = lambda d, E0, E1, logh, logC: self._model(d, E0, E1, np.exp(logh), np.exp(logC))

        self.jacobian_function = lambda d, E0, E1, logh, logC: self._model_jacobian(d, E0, E1, logh, logC)

        self.bounds = tuple(zip(self.E0_bounds, self.Emax_bounds, self.logh_bounds, self.logC_bounds))

    def E(self, d):
        """Evaluate this model at dose d. If the model is not parameterized, returns 0.

        Parameters
        ----------
        d : array_like
            Doses to calculate effect at
        
        Returns
        ----------
        effect : array_like
            Evaluate's the model at dose in d
        """
        if not self._is_parameterized():
            return super().E(d)

        return self._model(d, self.E0, self.Emax, self.h, self.C)

    def E_inv(self, E):
        """Inverse of the Hill equation

        Parameters
        ----------
        E : array_like
            Effects to get the doses for
        
        Returns
        ----------
        doses : array_like
            Doses which achieve effects E using this model. Effects that are
            outside the range [E0, Emax] will return np.nan for the dose
        """
        if not self._is_parameterized():
            return super().E_inv(E)

        return self._model_inv(E, self.E0, self.Emax, self.h, self.C)

    def get_parameters(self):
        """Gets the model's parmaters.

        Returns
        ----------
        parameters : tuple
            (E0, Emax, h, C)
        """
        return (self.E0, self.Emax, self.h, self.C)

    def _set_parameters(self, popt):
        E0, Emax, h, C = popt
        
        self.E0 = E0
        self.Emax = Emax
        self.h = h
        self.C = C
        
    def _model(self, d, E0, Emax, h, C):
        dh = np.power(d,h)
        return E0 + (Emax-E0)*dh/(np.power(C,h)+dh)

    def _model_inv(self, E, E0, Emax, h, C):
        E_ratio = (E-E0)/(Emax-E)
        d = np.float_power(E_ratio, 1./h)*C
        if hasattr(E,"__iter__"):
            d[E_ratio<0] = np.nan
            return d
        elif d < 0: return np.nan
        return d

    def _model_jacobian(self, d, E0, Emax, logh, logC):
        """
        Returns
        ----------
        jacobian : array_like
            Derivatives of the Hill equation with respect to E0, Emax, logh,
            and logC
        """
        
        dh = d**(np.exp(logh))
        Ch = (np.exp(logC))**(np.exp(logh))
        logd = np.log(d)

        jE0 = 1 - dh/(Ch+dh)
        jEmax = 1-jE0

        jC = (E0-Emax)*dh*np.exp(logh+logC)*(np.exp(logC))**(np.exp(logh)-1) / ((Ch+dh)*(Ch+dh))

        jh = (Emax-E0)*dh*np.exp(logh) * ((Ch+dh)*logd - (logC*Ch + logd*dh)) / ((Ch+dh)*(Ch+dh))
        
        jac = np.hstack((jE0.reshape(-1,1), jEmax.reshape(-1,1), jh.reshape(-1,1), jC.reshape(-1,1)))
        jac[np.isnan(jac)]=0
        return jac

    def _get_initial_guess(self, d, E, p0=None):

        if p0 is None:
            p0 = [max(E), min(E), 1, np.median(d)]

        p0 = list(self._transform_params_to_fit(p0))
        sanitize_initial_guess(p0, self.bounds)
        return p0

    def _transform_params_from_fit(self, params):
        return params[0], params[1], np.exp(params[2]), np.exp(params[3])

    def _transform_params_to_fit(self, params):
        return params[0], params[1], np.log(params[2]), np.log(params[3])

    def create_fit(d, E, E0_bounds=(-np.inf, np.inf), Emax_bounds=(-np.inf, np.inf), h_bounds=(0,np.inf), C_bounds=(0,np.inf), **kwargs):
        """Courtesy function to build a Hill model directly from data.
        Initializes a model using the provided bounds, then fits.
        """
        drug = Hill(E0_bounds=E0_bounds, Emax_bounds=Emax_bounds, h_bounds=h_bounds, C_bounds=C_bounds)
        drug.fit(d, E, **kwargs)
        return drug

    def __repr__(self):
        if not self._is_parameterized(): return "Hill()"
        
        return "Hill(E0=%0.2f, Emax=%0.2f, h=%0.2f, C=%0.2e)"%(self.E0, self.Emax, self.h, self.C)


class Hill_2P(Hill):
    """The two-parameter Hill equation

                            d^h
    E = E0 + (Emax-E0) * ---------
                         C^h + d^h

    Mathematically equivalent to the four-parameter Hill equation, but E0 and Emax are held constant (not fit to data).
    
    """
    def __init__(self, h=None, C=None, h_bounds=(0,np.inf), C_bounds=(0,np.inf), E0=1, Emax=0, **kwargs):
        super().__init__(h=h, C=C, E0=E0, Emax=Emax, h_bounds=h_bounds, C_bounds=C_bounds)

        self.fit_function = lambda d, logh, logC: self._model(d, self.E0, self.Emax, np.exp(logh), np.exp(logC))

        self.jacobian_function = lambda d, logh, logC: self._model_jacobian(d, logh, logC)

        self.bounds = tuple(zip(self.logh_bounds, self.logC_bounds))

    def _model_jacobian(self, d, logh, logC):
        dh = d**(np.exp(logh))
        Ch = (np.exp(logC))**(np.exp(logh))
        logd = np.log(d)
        E0 = self.E0
        Emax = self.Emax

        jC = (E0-Emax)*dh*np.exp(logh+logC)*(np.exp(logC))**(np.exp(logh)-1) / ((Ch+dh)*(Ch+dh))

        jh = (Emax-E0)*dh*np.exp(logh) * ((Ch+dh)*logd - (logC*Ch + logd*dh)) / ((Ch+dh)*(Ch+dh))
        
        jac = np.hstack((jh.reshape(-1,1), jC.reshape(-1,1)))
        jac[np.isnan(jac)]=0
        return jac

    def _get_initial_guess(self, d, E, p0=None):

        if p0 is None:
            p0 = [1, np.median(d)]
            
        p0 = list(self._transform_params_to_fit(p0))
        sanitize_initial_guess(p0, self.bounds)
        
        return p0

    def create_fit(d, E, E0=1, Emax=0, h_bounds=(0,np.inf), C_bounds=(0,np.inf), **kwargs):
        drug = Hill_2P(E0=E0, Emax=Emax, h_bounds=h_bounds, C_bounds=C_bounds)
        drug.fit(d, E, **kwargs)
        return drug

    def get_parameters(self):
        """Gets the model's parameters
        
        Returns
        ----------
        parameters : tuple
            (h, C)
        """
        return (self.h, self.C)

    def _set_parameters(self, popt):
        h, C = popt
        
        self.h = h
        self.C = C

    def _transform_params_from_fit(self, params):
        return np.exp(params[0]), np.exp(params[1])

    def _transform_params_to_fit(self, params):
        return np.log(params[0]), np.log(params[1])

    def __repr__(self):
        if not self._is_parameterized(): return "Hill_2P()"
        
        return "Hill_2P(E0=%0.2f, Emax=%0.2f, h=%0.2f, C=%0.2e)"%(self.E0, self.Emax, self.h, self.C)
    
class Hill_CI(Hill_2P):
    """Mathematically equivalent two-parameter Hill equation with E0=1 and Emax=0. However, Hill_CI.fit() uses the log-linearization approach to dose-response fitting used by the Combination Index.
    """
    def __init__(self, h=None, C=None, **kwargs):
        super().__init__(h=h, C=C, E0=1., Emax=0.)


    def _internal_fit(self, d, E, use_jacobian, **kwargs):
        mask = np.where((E < 1) & (E > 0) & (d > 0))
        E = E[mask]
        d = d[mask]
        fU = E
        fA = 1-E

        median_effect_line = linregress(np.log(d),np.log(fA/fU))
        h = median_effect_line.slope
        C = np.exp(-median_effect_line.intercept / h)
        
        return (h, C)

    def create_fit(d, E):
        drug = Hill_CI()
        drug.fit(d, E)
        return drug

    def plot_linear_fit(self, d, E, ax=None):
        if not self._is_parameterized():
            # TODO: Error
            return

        try:
            from matplotlib import pyplot as plt
        except:
            # TODO: Error
            # TODO: Move this whole function to plot
            return
        mask = np.where((E < 1) & (E > 0) & (d > 0))
        E = E[mask]
        d = d[mask]
        fU = E
        fA = 1-E

        ax_created = False
        if ax is None:
            ax_created = True
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111)

        ax.scatter(np.log(d), np.log(fA/fU))
        ax.plot(np.log(d), np.log(d)*self.h - self.h*np.log(self.C))

        if False:
            for i in range(self.bootstrap_parameters.shape[0]):
                h, C = self.bootstrap_parameters[i,:]
                ax.plot(np.log(d), np.log(d)*h - h*np.log(C), c='k', alpha=0.1, lw=0.5)

        ax.set_ylabel("h*log(d) - h*log(C)")
        ax.set_xlabel("log(d)")
        ax.set_title("CI linearization")
        if (ax_created):
            plt.tight_layout()
            plt.show()

    def _bootstrap_resample(self, d, E, use_jacobian, bootstrap_iterations, confidence_interval, **kwargs):
        """Bootstrap resampling is not yet implemented for CI
        """
        pass

    def __repr__(self):
        if not self._is_parameterized(): return "Hill_CI()"
        
        return "Hill_CI(h=%0.2f, C=%0.2e)"%(self.h, self.C)


def sham(d, drug):
    """Simulates a sham combination experiment. In a sham experiment, the two drugs combined are (secretly) the same drug. For example, a sham combination may add 10uM drugA + 20uM drugB. But because drugA and drugB are the same (drugX), the combination is really just equivalent to 30uM of the drug.    
    """
    if not 0 in d:
        d = np.append(0,d)
    d1, d2 = np.meshgrid(d,d)
    d1 = d1.flatten()
    d2 = d2.flatten()
    E = drug.E(d1+d2)
    return d1, d2, E

def sham_higher(d, drug, n_drugs):
    """Simulates a sham combination experiment for 3+ drugs. In a sham experiment, the two drugs combined are (secretly) the same drug. For example, a sham combination may add 10uM drugA + 20uM drugB. But because drugA and drugB are the same (drugX), the combination is really just equivalent to 30uM of the drug.

    Parameters
    ----------
    d : array_like
        Dose escalation to use for each "sham" drug

    drug
        A parameterized drug model from synergy.single, such as Hill, Hill_2P, Hill_CI, or MarginalLinear.

    n_drugs : int
        The number of drugs to include in the sham combination

    Returns
    ----------
    doses : M x n_drugs numpy.ndarray
        All dose pairs for each combination. In total, there are M samples, taken from n_drugs drugs.

    E : numpy.array
        Sham effects calculated as drug.E(doses.sum(axis=1))
    """
    if not 0 in d:
        d = np.append(0,d)
    doses = [d, ]*n_drugs
    doses = list(np.meshgrid(*doses))
    for i in range(n_drugs):
        doses[i] = doses[i].flatten()
    doses = np.asarray(doses).T
    E = drug.E(doses.sum(axis=1))
    return doses, E

def remove_zeros(d, min_buffer=0.2):
    """Replace zeros with some semi-intelligently chosen small value

    When plotting on a log scale, 0 doses can cause problems. This replaces all 0's using the dilution factor between the smallest non-zero, and second-smallest non-zero doses. If that dilution factor is too close to 1, it will replace 0's doses with a dose that is min_buffer*(max(d)-min(d[d>0])) less than min(d[d>0]) on a log scale.

    Parameters
    ----------
    d : array_like
        Doses to remove zeros from. Original array will not be changed.

    min_buffer : float , default=0.2
        For very large dose arrays with very small step sizes (useful for getting smooth plots), replacing 0's may lead to a value too close to the smallest non-zero dose. min_buffer is the minimum buffer (in log scale, relative to the full dose range) that 0's will be replaced with.
    """

    d=np.array(d,copy=True)
    dmin = np.min(d[d>0]) # smallest nonzero dose
    dmin2 = np.min(d[d>dmin])
    dilution = dmin/dmin2

    dmax = np.max(d)
    logdmin = np.log(dmin)
    logdmin2 = np.log(dmin2)
    logdmax = np.log(dmax)

    if (logdmin2-logdmin) / (logdmax-logdmin) < min_buffer:
        logdmin2_effective = logdmin + min_buffer*(logdmax-logdmin)
        dilution = dmin/np.exp(logdmin2_effective)

    d[d==0]=dmin * dilution
    return d

def residual_ss(d1, d2, E, model):
    E_model = model(d1, d2)
    return np.sum((E-E_model)**2)

def residual_ss_1d(d, E, model):
    E_model = model(d)
    return np.sum((E-E_model)**2)

def AIC(sum_of_squares_residuals, n_parameters, n_samples):
    """
    SOURCE: AIC under the Framework of Least Squares Estimation, HT Banks, Michele L Joyner, 2017
    Equations (6) and (16)
    https://projects.ncsu.edu/crsc/reports/ftp/pdf/crsc-tr17-09.pdf
    """
    aic = n_samples * np.log(sum_of_squares_residuals / n_samples) + 2*(n_parameters + 1)
    if n_samples / n_parameters > 40:
        return aic
    else:
        return aic + 2*n_parameters*(n_parameters+1) / (n_samples - n_parameters - 1)

def BIC(sum_of_squares_residuals, n_parameters, n_samples):
    return n_samples * np.log(sum_of_squares_residuals / n_samples) + (n_parameters+1)*np.log(n_samples)

def r_squared(E, sum_of_squares_residuals):
    ss_tot = np.sum((E-np.mean(E))**2)
    return 1-sum_of_squares_residuals/ss_tot

def sanitize_initial_guess(p0, bounds):
    """
    Makes sure p0 is within the bounds
    """
    index = 0
    for x, lower, upper in zip(p0, *bounds):
        if x is None:
            if True in np.isinf((lower,upper)): np.min((np.max((0,lower)), upper))
            else: p0[index]=np.mean((lower,upper))

        elif x < lower: p0[index]=lower
        elif x > upper: p0[index]=upper
        index += 1

def sanitize_single_drug_model(model, default_class, expected_superclass=None, **kwargs):
    """
    Makes sure the given single drug model is a class or object of a class that is permissible for the given synergy model.

    Parameters
    ----------
    model : object or class
        A single drug model

    default_class : class
        The type of model to return if the given model is of the wrong type

    expected_superclass : class , default=None
        The class the model is expected to be an instance of

    Returns
    -------
    model : object
        An object that is an instance of expected_superclass
    """
    # The model is a class
    if inspect.isclass(model):

        # If there is no expected_superclass, assume the given class is fine
        if expected_superclass is None:
            return model(**kwargs)

        else:
            # The model is a subclass of the expected subclass
            if issubclass(model, expected_superclass):
                # We are good!
                return model(**kwargs)

            # The given class violates the expected class: return the default
            else:
                if model is not None: warnings.warn("Expected single drug model to be subclass of %s, instead got %s"%(expected_superclass, model))
                return default_class(**kwargs)
        
        return model(**kwargs)

    # The model is an object
    else:
        # There is no expected_superclass, so assume the object is fine
        if expected_superclass is None:
            if model is None:
                return default_class(**kwargs)
            return model
        
        # The model is an instance of the expected_superclass, so good!
        elif isinstance(model,expected_superclass):
            return model

        # The model is an instance of the wrong type of class, so return the default
        else:
            if model is not None: warnings.warn("Expected single drug model to be subclass of %s, instead got %s"%(expected_superclass, type(model)))

            return default_class(**kwargs)
    return default_class(**kwargs)

class MuSyC(ParametricModel):
    """Multidimensional Synergy of Combinations (MuSyC) is a drug synergy framework based on the law of mass action (doi: 10.1016/j.cels.2019.01.003, doi: 10.1101/683433). In MuSyC, synergy is parametrically defined as shifts in potency, efficacy, or cooperativity.

    alpha21 : float
        Synergistic potency ([0,1) = antagonism, (1,inf) = synergism).        At large concentrations of drug 2, the "effective dose" of drug 1 = alpha21*d1.
    
    alpha12 : float
        Synergistic potency ([0,1) = antagonism, (1,inf) = synergism).         At large concentrations of drug 1, the "effective dose" of drug 2 = alpha12*d2.

    beta : float
        Synergistic efficacy ((-inf,0) = antagonism, (0,inf) = synergism). At large concentrations of both drugs, the combination achieves an effect beta-% stronger (or weaker) than the stronger single-drug.

    gamma21 : float
        Synergistic cooperativity ([0,1) = antagonism, (1,inf) = synergism). At large concentrations of drug 2, the Hill slope of drug 1 = gamma21*h1

    gamma12 : float
        Synergistic cooperativity ([0,1) = antagonism, (1,inf) = synergism). At large concentrations of drug 1, the Hill slope of drug 2 = gamma12*h2

    """
    def __init__(self, h1_bounds=(0,np.inf), h2_bounds=(0,np.inf),  \
            C1_bounds=(0,np.inf), C2_bounds=(0,np.inf),             \
            E0_bounds=(-np.inf,np.inf), E1_bounds=(-np.inf,np.inf), \
            E2_bounds=(-np.inf,np.inf), E3_bounds=(-np.inf,np.inf), \
            alpha12_bounds=(0,np.inf), alpha21_bounds=(0,np.inf),   \
            gamma12_bounds=(0,np.inf), gamma21_bounds=(0,np.inf),   \
            r1r=1., r2r=1., E0=None, E1=None, E2=None, E3=None,     \
            h1=None, h2=None, C1=None, C2=None, alpha12=None,       \
            alpha21=None, gamma12=None, gamma21=None, variant="full"):
        super().__init__()
        self.C1_bounds = C1_bounds
        self.C2_bounds = C2_bounds
        self.h1_bounds = h1_bounds
        self.h2_bounds = h2_bounds
        self.E0_bounds = E0_bounds
        self.E1_bounds = E1_bounds
        self.E2_bounds = E2_bounds
        self.E3_bounds = E3_bounds
        self.alpha12_bounds = alpha12_bounds
        self.alpha21_bounds = alpha21_bounds
        self.gamma12_bounds = gamma12_bounds
        self.gamma21_bounds = gamma21_bounds

        self.variant = variant

        self.r1r = r1r
        self.r2r = r2r
        self.E0 = E0
        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.h1 = h1
        self.h2 = h2
        self.C1 = C1
        self.C2 = C2
        self.alpha12 = alpha12
        self.alpha21 = alpha21
        self.gamma12 = gamma12
        self.gamma21 = gamma21
        if not None in [E1, E2, E3]:
            self.beta = (min(E1,E2)-E3) / (E0 - min(E1,E2))
        else:
            self.beta = None

        with np.errstate(divide='ignore'):
            self.logh1_bounds = (np.log(h1_bounds[0]), np.log(h1_bounds[1]))
            self.logC1_bounds = (np.log(C1_bounds[0]), np.log(C1_bounds[1]))
            self.logh2_bounds = (np.log(h2_bounds[0]), np.log(h2_bounds[1]))
            self.logC2_bounds = (np.log(C2_bounds[0]), np.log(C2_bounds[1]))
            
            self.logalpha12_bounds = (np.log(alpha12_bounds[0]), np.log(alpha12_bounds[1]))
            self.logalpha21_bounds = (np.log(alpha21_bounds[0]), np.log(alpha21_bounds[1]))

            self.loggamma12_bounds = (np.log(gamma12_bounds[0]), np.log(gamma12_bounds[1]))
            self.loggamma21_bounds = (np.log(gamma21_bounds[0]), np.log(gamma21_bounds[1]))


        if variant == "full":
            self.fit_function = lambda d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21, loggamma12, loggamma21: self._model(d[0], d[1], E0, E1, E2, E3, np.exp(logh1), np.exp(logh2), np.exp(logC1), np.exp(logC2), self.r1r, self.r2r, np.exp(logalpha12), np.exp(logalpha21), np.exp(loggamma12), np.exp(loggamma21))

            self.jacobian_function = lambda d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21, loggamma12, loggamma21: jacobian(d[0], d[1], E0, E1, E2, E3, logh1, logh2, logC1, logC2, self.r1r, self.r2r, logalpha12, logalpha21, loggamma12, loggamma21)

            self.bounds = tuple(zip(self.E0_bounds, self.E1_bounds, self.E2_bounds, self.E3_bounds, self.logh1_bounds, self.logh2_bounds, self.logC1_bounds, self.logC2_bounds, self.logalpha12_bounds, self.logalpha21_bounds, self.loggamma12_bounds, self.loggamma21_bounds))
        
        elif variant == "no_gamma":
            self.fit_function = lambda d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21: self._model(d[0], d[1], E0, E1, E2, E3, np.exp(logh1), np.exp(logh2), np.exp(logC1), np.exp(logC2), self.r1r, self.r2r, np.exp(logalpha12), np.exp(logalpha21), 1, 1)

            self.jacobian_function = lambda d, E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21: jacobian(d[0], d[1], E0, E1, E2, E3, logh1, logh2, logC1, logC2, self.r1r, self.r2r, logalpha12, logalpha21, 0, 0)[:,:-2]

            self.bounds = tuple(zip(self.E0_bounds, self.E1_bounds, self.E2_bounds, self.E3_bounds, self.logh1_bounds, self.logh2_bounds, self.logC1_bounds, self.logC2_bounds, self.logalpha12_bounds, self.logalpha21_bounds))


    def _get_initial_guess(self, d1, d2, E, drug1_model, drug2_model, p0=None):
        

        # If there is no intial guess, use single-drug models to come up with intitial guess
        if p0 is None:
            # Sanitize single-drug models
            default_class, expected_superclass = self._get_single_drug_classes()

            drug1_model = sanitize_single_drug_model(drug1_model, default_class, expected_superclass=expected_superclass, E0_bounds=self.E0_bounds, Emax_bounds=self.E1_bounds, h_bounds=self.h1_bounds, C_bounds=self.C1_bounds)

            drug2_model = sanitize_single_drug_model(drug2_model, default_class, expected_superclass=expected_superclass, E0_bounds=self.E0_bounds, Emax_bounds=self.E2_bounds, h_bounds=self.h2_bounds, C_bounds=self.C2_bounds)

            # Fit the single drug models if they were not pre-fit by the user
            if not drug1_model.is_fit():
                mask = np.where(d2==min(d2))
                drug1_model.fit(d1[mask], E[mask])
            if not drug2_model.is_fit():
                mask = np.where(d1==min(d1))
                drug2_model.fit(d2[mask], E[mask])

            # Get initial guesses of E0, E1, E2, h1, h2, C1, and C2 from single-drug fits
            E0_1, E1, h1, C1 = drug1_model.get_parameters()
            E0_2, E2, h2, C2 = drug2_model.get_parameters()
            
            #TODO: E orientation
            # Get initial guess of E3 at E(d1_max, d2_max), if that point exists
            E3 = E[(d1==max(d1)) & (d2==max(d2))]
            if len(E3)>0: E3 = np.mean(E3)

            # Otherwise guess E3 is the minimum E observed
            else: E3 = np.min(E)
            
            p0 = [(E0_1+E0_2)/2., E1, E2, E3, h1, h2, C1, C2, 1, 1, 1, 1]
        
            if self.variant == "no_gamma":
                p0 = p0[:-2]

        p0 = list(self._transform_params_to_fit(p0))
        sanitize_initial_guess(p0, self.bounds)
        return p0

    def _transform_params_from_fit(self, params):
        
        if self.variant == "no_gamma":
            E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21 = params
        else:
            E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21, loggamma12, loggamma21 = params
            gamma12 = np.exp(loggamma12)
            gamma21 = np.exp(loggamma21)
        
        h1 = np.exp(logh1)
        h2 = np.exp(logh2)
        C1 = np.exp(logC1)
        C2 = np.exp(logC2)
        alpha12 = np.exp(logalpha12)
        alpha21 = np.exp(logalpha21)
        
        if self.variant == "no_gamma":
            return E0, E1, E2, E3, h1, h2, C1, C2, alpha12, alpha21
        
        return E0, E1, E2, E3, h1, h2, C1, C2, alpha12, alpha21, gamma12, gamma21

    def _transform_params_to_fit(self, params):
        
        if self.variant == "no_gamma":
            E0, E1, E2, E3, h1, h2, C1, C2, alpha12, alpha21 = params
        else:
            E0, E1, E2, E3, h1, h2, C1, C2, alpha12, alpha21, gamma12, gamma21 = params
            loggamma12 = np.log(gamma12)
            loggamma21 = np.log(gamma21)

        logh1 = np.log(h1)
        logh2 = np.log(h2)
        logC1 = np.log(C1)
        logC2 = np.log(C2)
        logalpha12 = np.log(alpha12)
        logalpha21 = np.log(alpha21)

        if self.variant == "no_gamma":
            return E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21

        return E0, E1, E2, E3, logh1, logh2, logC1, logC2, logalpha12, logalpha21, loggamma12, loggamma21

    def E(self, d1, d2):
        if not self._is_parameterized():
            return None

        if self.variant == "no_gamma":
            return self._model(d1, d2, self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.r1r, self.r2r, self.alpha12, self.alpha21, 1, 1)
        
        else:
            return self._model(d1, d2, self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.r1r, self.r2r, self.alpha12, self.alpha21, self.gamma12, self.gamma21)

    def _get_parameters(self):
        if self.variant == "no_gamma":
            return self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.alpha12, self.alpha21
        else:
            return self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.alpha12, self.alpha21, self.gamma12, self.gamma21
    
    def _set_parameters(self, popt):
        if self.variant == "no_gamma":
            self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.alpha12, self.alpha21 = popt
        else:
            self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.alpha12, self.alpha21, self.gamma12, self.gamma21 = popt

    def _get_single_drug_classes(self):
        return Hill, Hill

    def _C_to_r1(self, C, h, r1r):
        return r1r/np.power(C,h)

    @DeprecationWarning
    def _C_to_r1r(self, C, h, r1):
        return r1*C**h

    @DeprecationWarning
    def _r_to_C(self, h, r1r):
        return (r1r/r1)**(1./h)

    def _reference_E(self, d1, d2):
        if not self._is_parameterized():
            return None
        return self._model(d1, d2, self.E0, self.E1, self.E2, min(self.E1,self.E2), self.h1, self.h2, self.C1, self.C2, self.r1r, self.r2r, 1, 1, 1, 1)

    def _model(self, d1, d2, E0, E1, E2, E3, h1, h2, C1, C2, r1r, r2r, alpha12, alpha21, gamma12, gamma21):

        d1h1 = np.power(d1,h1)
        d2h2 = np.power(d2,h2)

        C1h1 = np.power(C1,h1)
        C2h2 = np.power(C2,h2)

        r1 = r1r/C1h1
        r2 = r2r/C2h2

        alpha21d1gamma21h1 = np.power(alpha21*d1, gamma21*h1)
        alpha12d2gamma12h2 = np.power(alpha12*d2, gamma12*h2)

        C12h1 = np.power(C1,2*h1)
        C22h2 = np.power(C2,2*h2)

        # ********** U ********

        U=(r1*r2*np.power((r1*C1h1),gamma21)*C1h1*C2h2+r1*r2*np.power((r2*C2h2),gamma12)*C1h1*C2h2+np.power(r1,(gamma21+1))*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)*C1h1+np.power(r2,(gamma12+1))*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)*C2h2)/(d1h1*r1*r2*np.power((r1*C1h1),gamma21)*C2h2+d1h1*r1*r2*np.power((r2*C2h2),gamma12)*C2h2+d1h1*r1*np.power(r2,(gamma12+1))*alpha12d2gamma12h2*C2h2+d1h1*r1*np.power(r2,gamma12)*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)+d1h1*np.power(r1,(gamma21+1))*np.power(r2,gamma12)*alpha21d1gamma21h1*alpha12d2gamma12h2+d1h1*np.power(r1,(gamma21+1))*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)+d2h2*r1*r2*np.power((r1*C1h1),gamma21)*C1h1+d2h2*r1*r2*np.power((r2*C2h2),gamma12)*C1h1+d2h2*np.power(r1,(gamma21+1))*r2*alpha21d1gamma21h1*C1h1+d2h2*np.power(r1,gamma21)*r2*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)+d2h2*np.power(r1,gamma21)*np.power(r2,(gamma12+1))*alpha21d1gamma21h1*alpha12d2gamma12h2+d2h2*np.power(r2,(gamma12+1))*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)+r1*r2*np.power((r1*C1h1),gamma21)*C1h1*C2h2+r1*r2*np.power((r2*C2h2),gamma12)*C1h1*C2h2+np.power(r1,(gamma21+1))*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)*C1h1+np.power(r2,(gamma12+1))*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)*C2h2)

        #**********E1********

        A1=(d1h1*r1*r2*np.power((r1*C1h1),gamma21)*C2h2+d1h1*r1*r2*np.power((r2*C2h2),gamma12)*C2h2+d1h1*np.power(r1,(gamma21+1))*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)+d2h2*np.power(r1,gamma21)*r2*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12))/(d1h1*r1*r2*np.power((r1*C1h1),gamma21)*C2h2+d1h1*r1*r2*np.power((r2*C2h2),gamma12)*C2h2+d1h1*r1*np.power(r2,(gamma12+1))*alpha12d2gamma12h2*C2h2+d1h1*r1*np.power(r2,gamma12)*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)+d1h1*np.power(r1,(gamma21+1))*np.power(r2,gamma12)*alpha21d1gamma21h1*alpha12d2gamma12h2+d1h1*np.power(r1,(gamma21+1))*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)+d2h2*r1*r2*np.power((r1*C1h1),gamma21)*C1h1+d2h2*r1*r2*np.power((r2*C2h2),gamma12)*C1h1+d2h2*np.power(r1,(gamma21+1))*r2*alpha21d1gamma21h1*C1h1+d2h2*np.power(r1,gamma21)*r2*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)+d2h2*np.power(r1,gamma21)*np.power(r2,(gamma12+1))*alpha21d1gamma21h1*alpha12d2gamma12h2+d2h2*np.power(r2,(gamma12+1))*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)+r1*r2*np.power((r1*C1h1),gamma21)*C1h1*C2h2+r1*r2*np.power((r2*C2h2),gamma12)*C1h1*C2h2+np.power(r1,(gamma21+1))*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)*C1h1+np.power(r2,(gamma12+1))*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)*C2h2)

        #**********E2********

        A2=(d1h1*r1*np.power(r2,gamma12)*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)+d2h2*r1*r2*np.power((r1*C1h1),gamma21)*C1h1+d2h2*r1*r2*np.power((r2*C2h2),gamma12)*C1h1+d2h2*np.power(r2,(gamma12+1))*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21))/(d1h1*r1*r2*np.power((r1*C1h1),gamma21)*C2h2+d1h1*r1*r2*np.power((r2*C2h2),gamma12)*C2h2+d1h1*r1*np.power(r2,(gamma12+1))*alpha12d2gamma12h2*C2h2+d1h1*r1*np.power(r2,gamma12)*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)+d1h1*np.power(r1,(gamma21+1))*np.power(r2,gamma12)*alpha21d1gamma21h1*alpha12d2gamma12h2+d1h1*np.power(r1,(gamma21+1))*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)+d2h2*r1*r2*np.power((r1*C1h1),gamma21)*C1h1+d2h2*r1*r2*np.power((r2*C2h2),gamma12)*C1h1+d2h2*np.power(r1,(gamma21+1))*r2*alpha21d1gamma21h1*C1h1+d2h2*np.power(r1,gamma21)*r2*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)+d2h2*np.power(r1,gamma21)*np.power(r2,(gamma12+1))*alpha21d1gamma21h1*alpha12d2gamma12h2+d2h2*np.power(r2,(gamma12+1))*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)+r1*r2*np.power((r1*C1h1),gamma21)*C1h1*C2h2+r1*r2*np.power((r2*C2h2),gamma12)*C1h1*C2h2+np.power(r1,(gamma21+1))*alpha21d1gamma21h1*np.power((r2*C2h2),gamma12)*C1h1+np.power(r2,(gamma12+1))*alpha12d2gamma12h2*np.power((r1*C1h1),gamma21)*C2h2)

        
        return U*E0 + A1*E1 + A2*E2 + (1-(U+A1+A2))*E3
    
    @staticmethod
    def _get_beta(E0, E1, E2, E3):
        """Calculates synergistic efficacy, a synergy parameter derived from E parameters.
        """
        strongest_E = np.amin(np.asarray([E1,E2]), axis=0)
        beta = (strongest_E-E3) / (E0 - strongest_E)
        return beta


    def get_parameters(self, confidence_interval=95):
        if not self._is_parameterized():
            return None
        
        #beta = (min(self.E1,self.E2)-self.E3) / (self.E0 - min(self.E1,self.E2))
        beta = MuSyC._get_beta(self.E0, self.E1, self.E2, self.E3)

        if self.converged and self.bootstrap_parameters is not None:
            parameter_ranges = self.get_parameter_range(confidence_interval=confidence_interval)
        else:
            parameter_ranges = None

        params = dict()
        params['E0'] = [self.E0, ]
        params['E1'] = [self.E1, ]
        params['E2'] = [self.E2, ]
        params['E3'] = [self.E3, ]
        params['h1'] = [self.h1, ]
        params['h2'] = [self.h2, ]
        params['C1'] = [self.C1, ]
        params['C2'] = [self.C2, ]
        params['beta'] = [beta, ]
        params['alpha12'] = [self.alpha12, ]
        params['alpha21'] = [self.alpha21, ]
        if self.variant != "no_gamma":
            params['gamma12'] = [self.gamma12, ]
            params['gamma21'] = [self.gamma21, ]

        if parameter_ranges is not None:
            params['E0'].append(parameter_ranges[:,0])
            params['E1'].append(parameter_ranges[:,1])
            params['E2'].append(parameter_ranges[:,2])
            params['E3'].append(parameter_ranges[:,3])
            params['h1'].append(parameter_ranges[:,4])
            params['h2'].append(parameter_ranges[:,5])
            params['C1'].append(parameter_ranges[:,6])
            params['C2'].append(parameter_ranges[:,7])
            params['alpha12'].append(parameter_ranges[:,8])
            params['alpha21'].append(parameter_ranges[:,9])
            if self.variant != "no_gamma":
                params['gamma12'].append(parameter_ranges[:,10])
                params['gamma21'].append(parameter_ranges[:,11])

            bsE0 = self.bootstrap_parameters[:,0]
            bsE1 = self.bootstrap_parameters[:,1]
            bsE2 = self.bootstrap_parameters[:,2]
            bsE3 = self.bootstrap_parameters[:,3]
            beta_bootstrap = MuSyC._get_beta(bsE0, bsE1, bsE2, bsE3)

            beta_bootstrap = np.percentile(beta_bootstrap, [(100-confidence_interval)/2, 50+confidence_interval/2])
            params['beta'].append(beta_bootstrap)    
        return params
    
    def summary(self, confidence_interval=95, tol=0.01):
        pars = self.get_parameters(confidence_interval=confidence_interval)
        if pars is None:
            return None
        
        ret = []
        keys = pars.keys()
        # beta
        for key in keys:
            if "beta" in key:
                l = pars[key]
                if len(l)==1:
                    if l[0] < -tol:
                        ret.append("%s\t%0.2f\t(<0) antagonistic"%(key, l[0]))
                    elif l[0] > tol:
                        ret.append("%s\t%0.2f\t(>0) synergistic"%(key, l[0]))
                else:
                    v = l[0]
                    lb,ub = l[1]
                    if v < -tol and lb < -tol and ub < -tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(<0) antagonistic"%(key, v,lb,ub))
                    elif v > tol and lb > tol and ub > tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(>0) synergistic"%(key, v,lb,ub))
        # alpha
        for key in keys:
            if "alpha" in key:
                l = pars[key]
                if len(l)==1:
                    if np.log10(l[0]) < -tol:
                        ret.append("%s\t%0.2f\t(<1) antagonistic"%(key, l[0]))
                    elif np.log10(l[0]) > tol:
                        ret.append("%s\t%0.2f\t(>1) synergistic"%(key, l[0]))
                else:
                    v = l[0]
                    lb,ub = l[1]
                    if np.log10(v) < -tol and np.log10(lb) < -tol and np.log10(ub) < -tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(<1) antagonistic"%(key, v,lb,ub))
                    elif np.log10(v) > tol and np.log10(lb) > tol and np.log10(ub) > tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(>1) synergistic"%(key, v,lb,ub))

        # gamma
        for key in keys:
            if "gamma" in key:
                l = pars[key]
                if len(l)==1:
                    if np.log10(l[0]) < -tol:
                        ret.append("%s\t%0.2f\t(<1) antagonistic"%(key, l[0]))
                    elif np.log10(l[0]) > tol:
                        ret.append("%s\t%0.2f\t(>1) synergistic"%(key, l[0]))
                else:
                    v = l[0]
                    lb,ub = l[1]
                    if np.log10(v) < -tol and np.log10(lb) < -tol and np.log10(ub) < -tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(<1) antagonistic"%(key, v,lb,ub))
                    elif np.log10(v) > tol and np.log10(lb) > tol and np.log10(ub) > tol:
                        ret.append("%s\t%0.2f\t(%0.2f,%0.2f)\t(>1) synergistic"%(key, v,lb,ub))
        if len(ret)>0:
            return "\n".join(ret)
        else:
            return "No synergy or antagonism detected with %d percent confidence interval"%(int(confidence_interval))

    def __repr__(self):
        if not self._is_parameterized(): return "MuSyC()"
        
        #beta = (min(self.E1,self.E2)-self.E3) / (self.E0 - min(self.E1,self.E2))
        beta = MuSyC._get_beta(self.E0, self.E1, self.E2, self.E3)

        if self.variant == "no_gamma":
            return "MuSyC(E0=%0.2f, E1=%0.2f, E2=%0.2f, E3=%0.2f, h1=%0.2f, h2=%0.2f, C1=%0.2e, C2=%0.2e, alpha12=%0.2f, alpha21=%0.2f, beta=%0.2f)"%(self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.alpha12, self.alpha21, beta)
        return "MuSyC(E0=%0.2f, E1=%0.2f, E2=%0.2f, E3=%0.2f, h1=%0.2f, h2=%0.2f, C1=%0.2e, C2=%0.2e, alpha12=%0.2f, alpha21=%0.2f, beta=%0.2f, gamma12=%0.2f, gamma21=%0.2f)"%(self.E0, self.E1, self.E2, self.E3, self.h1, self.h2, self.C1, self.C2, self.alpha12, self.alpha21, beta, self.gamma12, self.gamma21)

def grid(d1min, d1max, d2min, d2max, n_points1, n_points2, replicates=1, logscale=True, include_zero=False):
    replicates = int(replicates)

    if logscale:
        d1 = np.logspace(np.log10(d1min), np.log10(d1max), num=n_points1)
        d2 = np.logspace(np.log10(d2min), np.log10(d2max), num=n_points2)
    else:
        d1 = np.linspace(d1min, d1max, num=n_points1)
        d2 = np.linspace(d2min, d2max, num=n_points2)

    if include_zero and logscale:
        if d1min > 0:
            d1 = np.append(0, d1)
        if d2min > 0:
            d2 = np.append(0, d2)
    
    D1, D2 = np.meshgrid(d1,d2)
    D1 = D1.flatten()
    D2 = D2.flatten()

    D1 = np.hstack([D1,]*replicates)
    D2 = np.hstack([D2,]*replicates)

    return D1, D2

def grid_multi(dmin, dmax, npoints, logscale=True, include_zero=False):
    if not (len(dmin)==len(dmax) and len(dmin)==len(npoints)):
        return None
    doses = []
    for Dmin, Dmax, n in zip(dmin, dmax, npoints):
        if logscale:
            logDmin = np.log10(Dmin)
            logDmax = np.log10(Dmax)
            d = np.logspace(logDmin, logDmax, num=n)
        else:
            d = np.linspace(Dmin, Dmax, num=n)
        if include_zero and Dmin > 0:
            d = np.append(0, d)
        doses.append(d)
    dosegrid = np.meshgrid(*doses)
    
    if include_zero:
        total_length = np.prod([i+1 for i in npoints])
    else:
        total_length = np.prod(npoints)
    n = len(dmin)
    return_d = np.zeros((total_length, n))
    
    for i in range(n):
        return_d[:,i] = dosegrid[i].flatten()

    return return_d


def get_num_replicates(d1, d2):
    """Given 1d dose arrays d1 and d2, determine how many replicates of each unique combination are present

    Parameters:
    -----------
    d1 : array_like, float
        Doses of drug 1

    d2 : array_like, float
        Doses of drug 2

    Returns:
    -----------
    replicates : numpy.array
        Counts of each unique dose combination
    """
    return np.unique(np.asarray([d1,d2]), axis=1, return_counts=True)[1]

@DeprecationWarning
def remove_replicates(d1, d2):
    """Given 1d dose arrays d1 and d2, remove replicates. This is needed sometimes for plotting, since some plot functions expect a single d1, d2 -> E for each dose.

    Parameters:
    -----------
    d1 : array_like, float
        Doses of drug 1

    d2 : array_like, float
        Doses of drug 2

    Returns:
    -----------
    d1 : array_like, float
        Doses of drug 1 without replicates

    d2 : array_like, float
        Doses of drug 2 without replicates
    """
    d = np.asarray(list(set(zip(d1, d2))))
    return d[:,0], d[:,1]

def generate_3dsur_data(d1, d2, E, scatter_points, **kwargs):
    
    d1 = np.array(d1, copy=True, dtype=np.float64)
    d2 = np.array(d2, copy=True, dtype=np.float64)
    E = np.asarray(E)

    d1 = remove_zeros(d1)
    d2 = remove_zeros(d2)
    d1 = np.log10(d1)
    d2 = np.log10(d2)

    sorted_indices = np.lexsort((d1,d2))
    D1 = d1[sorted_indices]
    D2 = d2[sorted_indices]
    E = E[sorted_indices]

    # Replicates
    n_replicates = np.unique(get_num_replicates(D1,D2))
    if len(n_replicates)>1:
        raise ValueError("Expects the same number of replicates for each dose")
    n_replicates = n_replicates[0]
    
    if n_replicates != 1:
        aggfunc = kwargs.get('aggfunc', np.median)
        print(F'Number of replicates : {n_replicates}. aggregates the data using the {aggfunc.__name__}')
        E_agg = []
        for e2 in np.unique(D2):
            for e1 in np.unique(D1):
                ix = (D1==e1) & (D2==e2)
                E_agg.append(aggfunc(E[ix]))
        E = np.array(E_agg)
        D1 = np.unique(D1)
        D2 = np.unique(D2)

        d1, d2 = np.meshgrid(D1, D2)

        n_d1 = len(np.unique(D1))
        n_d2 = len(np.unique(D2))
        E = E.reshape(n_d2,n_d1)
    else:
        n_d1 = len(np.unique(d1))
        n_d2 = len(np.unique(d2))
        d1 = d1.reshape(n_d2,n_d1)
        d2 = d2.reshape(n_d2,n_d1)
        E = E.reshape(n_d2,n_d1)

    zmax = max(abs(np.nanmin(E[~np.isinf(E)])), abs(np.nanmax(E[~np.isinf(E)])))
    vmin = -zmax
    vmax = zmax

    d1scatter = np.array(scatter_points['drug1.conc'], copy=True, dtype=np.float64)
    d2scatter = np.array(scatter_points['drug2.conc'], copy=True, dtype=np.float64)

    zero_mask_1 = np.where(d1scatter <= 0)
    pos_mask_1 = np.where(d1scatter > 0)

    zero_mask_2 = np.where(d2scatter <= 0)
    pos_mask_2 = np.where(d2scatter > 0)

    d1scatter[zero_mask_1] = np.min(d1)
    d2scatter[zero_mask_2] = np.min(d2)
    d1scatter[pos_mask_1] = np.log10(d1scatter[pos_mask_1])
    d2scatter[pos_mask_2] = np.log10(d2scatter[pos_mask_2])
    
    data_to_plot = {
        'xs': d1scatter,
        'ys': d2scatter,
        'zs': scatter_points['effect'],
        'x': d1,
        'y': d2,
        'z': E,
        'vmin': vmin,
        'vmax': vmax
    }

    return data_to_plot

df = pd.read_csv("https://raw.githubusercontent.com/djwooten/synergy/master/datasets/sample_data_1.csv")
print('model running')
model = MuSyC(E0_bounds=(0,1), E1_bounds=(0,1), E2_bounds=(0,1), E3_bounds=(0,1))
model.fit(df['drug1.conc'], df['drug2.conc'], df['effect'], bootstrap_iterations=100)
# Requires plotly
print('generate data')
data = generate_3dsur_data(df['drug1.conc'], df['drug2.conc'], xlabel="Drug1", 	\
                           E = model.E(df['drug1.conc'], df['drug2.conc']),
                          ylabel="Drug2", zlabel="Effect", fname="plotly.html", \
                          scatter_points=df)
print(data['z'])
