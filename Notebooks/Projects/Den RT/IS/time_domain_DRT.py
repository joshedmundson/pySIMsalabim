"""Find System Lifetimes using Distribution of Relaxation Times (DRT) Fitting in the Time Domain"""
######### Package Imports #########################################################################

import matplotlib.pyplot as plt
# from pySIMsalabim.utils import device_parameters as utils_dev
import scipy.optimize as so
import numpy as np

######### Class Definitions #######################################################################

class DRT_Fit:
    """ A class that bundles together all results from fitting a Z_DRT curve to data

    Attributes
    ----------
    U : numpy.ndarray, shape (m,)
        Fitted coefficients [U_1, U_2, ..., U_m]
    tau : numpy.ndarray, shape (m,)
        Fitted lifetimes [tau_1, tau_2, ..., tau_m]
    offset : numpy.float64
        Fitted value for offset
    cost : numpy.float64
        The value of the cost function 
            F(x) = 0.5 * sum(rho(error_i(x)**2), i = 0, ..., n - 1)
        at the solution. By default, rho(z) = z. See scipy.optimize.least_squares for details
    m : int
        Number of lifetimes/coefficients used in Z_DRT, given by the length of U/tau
    
    Methods
    -------
    set_Z_DRT(t)
        Calculates Z_DRT(t) using the object's attributes (self.U, self.tau, self.offset)
        and sets self.Z_DRT equal to the result 
    """
    def __init__(self, U, tau, offset, cost, m):
        self.U = U
        self.tau = tau 
        self.offset = offset 
        self.cost = cost
        self.m = m 
        self.Z_DRT = None

    def set_Z_DRT(self, t):
        """ Sets self.Z_DRT = Z_DRT(t) using the objects attributes
        

        """
        self.Z_DRT = Z_DRT(t, self.U, self.tau)


######### Function Definitions ####################################################################

def Z_DRT(t, U, tau, offset=0):
    """Calculate the function 
        Z_DRT(t) = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2) + ... + U_m*exp(-t/tau_m) + offset
    
    Parameters
    ----------
    t : float or list/numpy.ndarray
        The time values over which the simulated impedance-adjacent experiment takes place
    U : list/numpy.ndarray with shape (m,)
        Coefficients for the exponential decay functions such that U[i] is the coefficient of 
        exp(-t/tau[i])
    tau : list/numpy.ndarray with shape (m,)
        The distribution of relaxation times
    offset : float (optional)
        Offest applied to the sum over exponential decay functions
        
    Returns 
    -------
    calculated_Z_DRT : numpy.ndarray with shape (n,)
        Z_DRT at all time values in t
    """
    
    # Check to make sure all input has been formatted as numpy arrays
    U = np.array(U)
    tau = np.array(tau)
    
    if not isinstance(t, list) and not isinstance(t, np.ndarray):
        t = np.array([t])
    elif not isinstance(t, np.ndarray):
        t = np.array(t)
    
    # Calculate Z_DRT(t) for all values in t
    calculated_Z_DRT = (U @ np.exp(-np.outer(1/tau, t))) + offset
    
    return calculated_Z_DRT


def fit_Z_DRT(t, y, U0, tau0, offset0=0, set_Z_DRT=True, **kwargs):
    """Fits a Z_DRT(t) curve to a function y(t)

    Uses a non-linear approach to fit 
        Z_DRT(t) = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2) + ... + U_m*exp(-t/tau_m) + offset
    to a function y by finding the optimal values for the parameters 
        U = [U_1, U_2, ..., U_m] 
        tau = [tau_1, tau_2, ..., tau_m]
        
    Parameters
    ----------
    t : (list/numpy.ndarray) with shape (n,)
        The time values over which the function y is known
    y : list/numpy.ndarray with shape (n,)
        The values of y(t) that Z_DRT will be fitted to
    U0 : list/numpy.ndarray with shape (m,)
        Initial guess of U values for fitting 
    tau0 : list/numpy.ndarray with shape (m,)
        Initial guess of tau values for fitting
    offset0 : float (optional)
        Initial guess of the Z_DRT offset value
    set_Z_DRT : bool (optional)
        If true, will set the Z_DRT attribute of the returned DRT_Fit object using t
    kwargs
        Keyword arguments passed to scipy.optimize.least_squares for fitting
    
    Returns
    -------
    drt_fit : DRT_Fit 
        With attributes as follows
            U : numpy.ndarray, shape (m,)
                Fitted [U_1, U_2, ..., U_m] array
            tau : numpy.ndarray, shape (m, )
                Fitted [tau_1, tau_2, ..., tau_m] array
            offset : numpy.float64
                Fitted value for offset
            cost : numpy.float64
                The value of the cost function 
                    F(x) = 0.5 * sum(rho(error_i(x)**2), i = 0, ..., n - 1)
                at the solution. By default, rho(z) = z. See scipy.optimize.least_squares for details
            m : int
                Number of lifetimes/coefficients used in Z_DRT, given by the length of U/tau
    """
    
    # Make sure passed parameters are numpy.ndarrays
    t = np.array(t)
    y = np.array(y)
    U0 = np.array(U0)
    tau0 = np.array(tau0)
    m = len(U0)
    
    # Create an initial parameter array to pass to the fitting algorithm
    initial_params = np.concatenate((U0, tau0, [offset0]))
    
    # Create the error function 
    error = lambda x : y - Z_DRT(t, x[:m], x[m:-1], offset=x[-1])
    
    # Fit the params using scipy. Note x_scale='jac' makes for better fits than the default value
    fit = so.least_squares(error, x0=initial_params, x_scale='jac', **kwargs)
    
    # Store the fit results
    drt_fit = DRT_Fit(fit.x[:m], fit.x[m:-1], fit.x[-1], fit.cost, m)

    # Set the Z_DRT curves for each DRT_Fit result if calc_Z_DRT=True
    drt_fit.set_Z_DRT(t) if set_Z_DRT else None
    
    return drt_fit


def multi_fit_Z_DRT(t, y, m_values, set_Z_DRT=True, **kwargs):
    """ Runs fit_Z_DRT over multiple m values 
    
    Parameters
    ----------
    t : list or numpy.ndarray, shape (n,)
        Time values over which y(t) is known
    y : list or numpy.ndarray, shape(n, )
        y(t) values each Z_DRT curve is fitted to
    m_values : list, shape (i,)
        Various m values for fit_Z_DRT, where m is the number of coefficients/lifetimes
        in Z_DRT(t)
    set_Z_DRT : bool (optional)
        If true, will set the Z_DRT attribute of the returned DRT_Fit objects using t
    kwargs 
        Keyword arguments passed to scipy.optimize.least_squares for fitting

    Returns
    -------
    fits : list of DRT_Fit, shape (i,)
        List of DRT_Fit objects, one for each passed m value
    """
    fits = []
    
    # Fit the params for each m
    for m in m_values:
        # Default estimate U, tau, and offset 
        U0 = np.zeros(m)
        tau0 = np.geomspace(t[0], t[-1], m)
        offset0 = 0
        
        # Fit the curve
        fit = fit_Z_DRT(t, y, U0, tau0, offset0=offset0, set_Z_DRT=set_Z_DRT, **kwargs)
        
        # Add to fits
        fits.append(fit)
        
    return fits
        
        
        