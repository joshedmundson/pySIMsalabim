"""Find System Lifetimes using Distribution of Relaxation Times (DRT) Fitting in the Time Domain"""
######### Package Imports #########################################################################

import matplotlib.pyplot as plt
# from pySIMsalabim.utils import device_parameters as utils_dev
import scipy.optimize as so
import numpy as np
import torch
import osqp
from scipy.sparse import csc_matrix

######### References ##############################################################################

# [1] M. Schönleber, D. Klotz, and E. Ivers-Tiffée, ‘A Method for Improving the Robustness of linear 
# Kramers-Kronig Validity Tests’, Electrochimica Acta, vol. 131, pp. 20–27, June 2014, 
# doi: 10.1016/j.electacta.2014.01.034.

# [2] B. Stellato, G. Banjac, P. Goulart, A. Bemporad, and S. Boyd, ‘OSQP: an operator splitting solver 
# for quadratic programs’, Math. Prog. Comp., vol. 12, no. 4, pp. 637–672, Dec. 2020, 
# doi: 10.1007/s12532-020-00179-2.


######### Class Definitions #######################################################################

class DRT_Fit_Result:
    """ A class that bundles together all results from fitting a predict_y curve to data

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
        Number of lifetimes/coefficients used in predict_y, given by the length of U/tau
    
    Methods
    -------
    predict_y(t)
        Calculates predict_y(t) using the object's attributes (self.U, self.tau, self.offset)
        and sets self.predict_y equal to the result 
    """
    def __init__(self, U, tau, offset, m, MSE=None, R2=None):
        self.U = U
        self.tau = tau 
        self.offset = offset 
        self.m = m 
        self.MSE = MSE
        self.R2 = R2
        self.y = None

    def predict_y(self, t, backend='numpy', device=torch.device('cpu')):
        """ Sets self.predict_y = predict_y(t) using the objects attributes

        Parameters
        ----------
        t : float or list/numpy.ndarray, shape (n,)
            Time values to calculate the predict_y curve
        backend : {'numpy', 'torch'} (optional)
            Determines whether matrix operations are done with numpy or pytorch. Default numpy. 
        device : torch.device (optional)
            Determines what device is used for matrix ops if backend='torch'.
        
        Returns
        -------
        None
        """
        self.y = predict_y(t, self.U, self.tau, self.offset)

######### Function Definitions ####################################################################

# Utility Functions #####################################
def R2_error(y, y_model):
    SSres = np.sum((y - y_model)**2)
    SStot = np.sum((y - np.mean(y_model))**2)
    return 1 - SSres/SStot

def mean_square_error(y, y_model):
    return np.mean((y-y_model)**2)

def predict_y(t, U, tau, offset=0):
    """Use numpy.ndarray objects to calculate the function 
        predict_y(t) = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2) + ... + U_m*exp(-t/tau_m) + offset
    
    Parameters
    ----------
    t : numpy.ndarray, shape (n,)
        The time values over which the simulated impedance-adjacent experiment takes place
    U : numpy.ndarray with shape (m,)
        Coefficients for the exponential decay functions such that U[i] is the coefficient of 
        exp(-t/tau[i])
    tau : numpy.ndarray with shape (m,)
        The distribution of relaxation times
    offset : float (optional)
        Initial guess of the predict_y offset value
    device : torch.device 
        The specified device for torch based computation if backend='torch'. 
    
    Returns 
    -------
    predicted_y : numpy.ndarray with shape (n,)
        predict_y at all time values in t
    """
    # Calculate predict_y(t) for all values in t
    predicted_y = (U @ np.exp(-np.outer(1/tau, t))) + offset

    return predicted_y

def calculate_tau(time):
    m = len(time)

    # Find the smallest time interval in the array
    d_time = [time[i+1] - time[i] for i in range(len(time)-1)]
    time_min = np.min(d_time)
    time_max = time[-1]-time[0]

    tau_min = time_min/np.pi
    tau_max = time_max/(2*np.pi)

    tau_values = np.geomspace(tau_min, tau_max, m)

    return tau_values


# Fitting Functions #####################################
def osqp_linear_fit(time, y, tau, offset='Auto', scaling=True, bounds=None):
    """
    Converts the linear fit problem to a convex quadratic program and minimises using 
    the Operator Splitting Quadratic Program (OSQP) package solver [2].
    """
    # Record offset from y=0
    offset = y[-1] if offset == 'Auto' else offset

    # Step 0: Scale the signal and remove scaled offset
    scale_factor = 1
    if scaling:
        y_max = y.max()
        y_min = y.min()
        scale_factor = y_max-y_min
        y = (y - y_min)/scale_factor
        y = y - y[-1]
    
    # Step 1: Convert the DRT function form into the form Y = R@U
    m = len(tau)
    n = len(time)
    R = np.exp(-np.outer(1/tau, time)).T 
    
    # Step 2: Convert the least squares problem into a quadratic program
    P = 2*R.T@R 
    P = csc_matrix((1/2)*(P.T + P))
    q = -2*R.T@y
    
    # Step 3: Set the constraints matrices
    A = csc_matrix(np.identity(m))
    ones = np.ones((m, 1))
    l = bounds[0]*ones if bounds is not None else -np.inf*ones
    u = bounds[1]*ones if bounds is not None else np.inf*ones
    
    # Step 4: Solve 
    osqp_model = osqp.OSQP()
    osqp_model.setup(P, q, A, l, u, verbose=False)
    osqp_result = osqp_model.solve()
    
    # Step 5: Rescale results
    U = scale_factor*osqp_result.x
    y_model = R@U + offset
    y = y*scale_factor + offset

    # Step 6: Calculate errors
    MSE = mean_square_error(y, y_model)
    R2 = R2_error(y, y_model)

    # Step 7: Package results in DRT_Fit_result
    fit = DRT_Fit_Result(U, tau, m=m, offset=offset, MSE=MSE, R2=R2)
    fit.y = y_model
    
    return fit

# def osqp_checkerboard_fit(time, y, tau, offset='Auto', checkerboard_iters=100):
    
#     # Step 1: Prep the signal for fitting by removing offset and scaling
#     offset = y[-1] if offset == 'Auto' else offset
#     y_max = y.max()
#     y_min = y.min()
#     scale_factor = y_max-y_min
#     y_scaled = (y - y_min)/scale_factor
#     y_scaled = y_scaled - y_scaled[-1]

#     # Step 2: Set initial values
#     U_values = np.zeros(len(tau))
    
#     y_cap = y_scaled

#     cap_U_values = np.zeros(len(tau))
#     ind_U_values = np.zeros(len(tau))

#     cap_offset = 0
#     ind_offset = 0

#     fits = []
    
#     # NOTE: we probably want to set the offset and scale factor guesses ourselves
#     for i in range(checkerboard_iters):
        
#         # Set scale params for capacitive effects and fit
#         cap_fit = osqp_linear_fit(time, y_cap, tau, offset=cap_offset, scaling=False, bounds=(0, np.inf))
        
#         # Add the fit to U_values
#         cap_U_values = cap_fit.U
        
#         # Remove the capacitive effects from y
#         y_ind = y_scaled - cap_fit.y
#         ind_offset = y_ind[-1]
        
#         # Set the inductive scale params and fit by doing a capacitive fit on an inverted function
#         ind_fit = osqp_linear_fit(time, -y_ind, tau, offset=ind_offset, scaling=False, bounds=(0, np.inf))
        
#         # Add the fit to U_values 
#         ind_U_values = -ind_fit.U

#         # Remove the inductive effects from the curve for the next iteration
#         y_cap = y_scaled + ind_fit.y
#         cap_offset = y_cap[-1]
        
#         # Once the fit is done, return a fit object with the results and a dummy cost of 0
#         U_values = cap_U_values + ind_U_values

#         # Rescale and package results
#         U_values = scale_factor*U_values
#         y_model = predict_y(time, U_values, tau, offset=offset)
#         MSE = mean_square_error(y, y_model)
#         R2 = R2_error(y, y_model)

#         fit = DRT_Fit_Result(U_values, tau, offset, len(tau), MSE=MSE, R2=R2)
#         fit.y = y_model
#         fits.append(fit)
        
#     return fits

def osqp_checkerboard_fit(time, y, tau, offset='Auto', checkerboard_iters=100):
    
    # Step 1: Prep the signal for fitting by removing offset and scaling
    offset = y[-1] if offset == 'Auto' else offset

    # Step 2: Set initial values
    U_values = np.zeros(len(tau))
    
    y_cap = y

    cap_U_values = np.zeros(len(tau))
    ind_U_values = np.zeros(len(tau))

    cap_offset = 0
    ind_offset = 0

    fits = []
    
    # NOTE: we probably want to set the offset and scale factor guesses ourselves
    for i in range(checkerboard_iters):
        
        # Set scale params for capacitive effects and fit
        cap_fit = osqp_linear_fit(time, y_cap, tau, offset=cap_offset, bounds=(0, np.inf))
        
        # Add the fit to U_values
        cap_U_values = cap_fit.U
        
        # Remove the capacitive effects from y
        y_ind = y - cap_fit.y
        ind_offset = y_ind[-1]
        
        # Set the inductive scale params and fit by doing a capacitive fit on an inverted function
        ind_fit = osqp_linear_fit(time, -y_ind, tau, offset=ind_offset, bounds=(0, np.inf))
        
        # Add the fit to U_values 
        ind_U_values = -ind_fit.U

        # Remove the inductive effects from the curve for the next iteration
        y_cap = y + ind_fit.y
        cap_offset = y_cap[-1]
        
        # Once the fit is done, return a fit object with the results and a dummy cost of 0
        U_values = cap_U_values + ind_U_values

        # Rescale and package results
        y_model = predict_y(time, U_values, tau, offset=offset)
        MSE = mean_square_error(y, y_model)
        R2 = R2_error(y, y_model)

        fit = DRT_Fit_Result(U_values, tau, offset, len(tau), MSE=MSE, R2=R2)
        fit.y = y_model
        fits.append(fit)
        
    return fits