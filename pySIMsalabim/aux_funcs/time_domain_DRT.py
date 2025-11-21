"""Find System Lifetimes using Distribution of Relaxation Times (DRT) Fitting in the Time Domain"""
######### Package Imports #########################################################################

import matplotlib.pyplot as plt
# from pySIMsalabim.utils import device_parameters as utils_dev
import scipy.optimize as so
import numpy as np

######### Class Definitions #######################################################################

class DRT_Fit_Result:
    """ A class that bundles together all results from fitting a DRT_curve curve to data

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
        Number of lifetimes/coefficients used in DRT_curve, given by the length of U/tau
    
    Methods
    -------
    set_DRT_curve(t)
        Calculates DRT_curve(t) using the object's attributes (self.U, self.tau, self.offset)
        and sets self.DRT_curve equal to the result 
    """
    def __init__(self, U, tau, offset, cost, m):
        self.U = U
        self.tau = tau 
        self.offset = offset 
        self.cost = cost
        self.m = m 
        self.DRT_curve = None

    def set_DRT_curve(self, t):
        """ Sets self.DRT_curve = DRT_curve(t) using the objects attributes

        Parameters
        ----------
        t : float or list/numpy.ndarray, shape (n,)
            Time values to calculate the DRT_curve curve
        
        Returns
        -------
        None
        """
        self.DRT_curve = DRT_curve(t, self.U, self.tau, self.offset)


######### Function Definitions ####################################################################

def DRT_curve(t, U, tau, offset=0):
    """Calculate the function 
        DRT_curve(t) = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2) + ... + U_m*exp(-t/tau_m) + offset
    
    Parameters
    ----------
    t : float or list/numpy.ndarray, shape (n,)
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
    calculated_DRT_curve : numpy.ndarray with shape (n,)
        DRT_curve at all time values in t
    """
    
    # Check to make sure all input has been formatted as numpy arrays
    U = np.array(U)
    tau = np.array(tau)
    
    if not isinstance(t, list) and not isinstance(t, np.ndarray):
        t = np.array([t])
    elif not isinstance(t, np.ndarray):
        t = np.array(t)
    
    # Calculate DRT_curve(t) for all values in t
    calculated_DRT_curve = (U @ np.exp(-np.outer(1/tau, t))) + offset
    
    return calculated_DRT_curve


def fit_DRT_curve(t, y, U0, tau0, offset0=0, set_DRT_curve=True, **kwargs):
    """Fits an DRT_curve(t) curve to a function y(t)

    Uses a non-linear approach to fit 
        DRT_curve(t) = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2) + ... + U_m*exp(-t/tau_m) + offset
    to a function y by finding the optimal values for the parameters 
        U = [U_1, U_2, ..., U_m] 
        tau = [tau_1, tau_2, ..., tau_m]
        
    Parameters
    ----------
    t : (list/numpy.ndarray) with shape (n,)
        The time values over which the function y is known
    y : list/numpy.ndarray with shape (n,)
        The values of y(t) that DRT_curve will be fitted to
    U0 : list/numpy.ndarray with shape (m,)
        Initial guess of U values for fitting 
    tau0 : list/numpy.ndarray with shape (m,)
        Initial guess of tau values for fitting
    offset0 : float (optional)
        Initial guess of the DRT_curve offset value
    set_DRT_curve : bool (optional)
        If true, will set the DRT_curve attribute of the returned DRT_Fit_Result object using t
    kwargs
        Keyword arguments passed to scipy.optimize.least_squares for fitting
    
    Returns
    -------
    drt_fit : DRT_Fit_Result 
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
                Number of lifetimes/coefficients used in DRT_curve, given by the length of U/tau
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
    error = lambda x : y - DRT_curve(t, x[:m], x[m:-1], offset=x[-1])
    
    # Fit the params using scipy. Note x_scale='jac' makes for better fits than the default value
    fit = so.least_squares(error, x0=initial_params, x_scale='jac', **kwargs)
    
    # Store the fit results
    drt_fit = DRT_Fit_Result(fit.x[:m], fit.x[m:-1], fit.x[-1], fit.cost, m)

    # Set the DRT_curve curves for each DRT_Fit_Result result if calc_DRT_curve=True
    drt_fit.set_DRT_curve(t) if set_DRT_curve else None
    
    return drt_fit


def multi_fit_DRT_curve(t, y, m_values, U_scale_factor=0, offset0=0, set_DRT_curve=True, **kwargs):
    """ Runs fit_DRT_curve over multiple m values 
    
    Parameters
    ----------
    t : list or numpy.ndarray, shape (n,)
        Time values over which y(t) is known
    y : list or numpy.ndarray, shape(n, )
        y(t) values each DRT_curve curve is fitted to
    m_values : list, shape (i,)
        Various m values for fit_DRT_curve, where m is the number of coefficients/lifetimes
        in DRT_curve(t)
    U_scale_factor : float
        Controls the scale for the intial guess of the coefficients U0 passed to fit_DRT_curve
    offset0 : float 
        Initial guess for the offset passed to fit_DRT_curve
    set_DRT_curve : bool (optional)
        If true, will set the DRT_curve attribute of the returned DRT_Fit_Result objects using t
    kwargs 
        Keyword arguments passed to scipy.optimize.least_squares for fitting

    Returns
    -------
    fits : list of DRT_Fit_Result, shape (i,)
        List of DRT_Fit_Result objects, one for each passed m value
    """
    fits = []
    
    # Fit the params for each m
    for m in m_values:
        # Default estimate U, tau, and offset 
        U0 = np.ones(m)*U_scale_factor
        tau0 = np.geomspace(t[0], t[-1], m)
        offset0 = offset0
        
        # Fit the curve
        fit = fit_DRT_curve(t, y, U0, tau0, offset0=offset0, set_DRT_curve=set_DRT_curve, **kwargs)
        
        # Add to fits
        fits.append(fit)
        
    return fits
        
        
        