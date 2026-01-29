"""Find System Lifetimes using Distribution of Relaxation Times (DRT) Fitting in the Time Domain"""
######### Package Imports #########################################################################

import matplotlib.pyplot as plt
# from pySIMsalabim.utils import device_parameters as utils_dev
import scipy.optimize as so
import argparse
import sys
import os
import traceback
import pickle
import pandas as pd
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
    offset : float
        Offset of the steady state of the fitted signal from 0
    m : int
        Number of lifetimes/coefficients used in predict_y, given by the length of U/tau
    y : {None, numpy.ndarray}
        Model given by
            y = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2)... + U_m*exp(-t/tau_m) + offset
    norm_U : {None, numpy.ndarray with shape (m,)}
        Fitted coefficients normalised by np.sum(self.U)
    MSE : float 
        Mean square error between the model (self.y) and the fitted signal
    R2 : float 
        R^2 error between the model (self.y) and the fitted signal

        
    Methods
    -------
    predict_y(t)
        Calculates predict_y(t) using the object's attributes (self.U, self.tau, self.offset)
        and sets self.predict_y equal to the result 
    set_norm_U()
        Sets self.norm_U to U/np.sum(U)
    """
    def __init__(self, U, tau, offset, m, y=None, MSE=None, R2=None):
        self.U = U
        self.tau = tau 
        self.offset = offset 
        self.m = m 
        self.y = None
        self.MSE = MSE
        self.R2 = R2
        self.norm_U = None

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

    def set_norm_U(self):
        """
        Set self.norm_U equal to self.U/np.sum(self.U)
        """
        self.norm_U = self.U/np.sum(self.U)

######### Function Definitions ####################################################################

# Utility Functions #####################################
def predict_y(time, U, tau, offset=0):
    """Calculate the function 
        predict_y(t) = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2) + ... + U_m*exp(-t/tau_m) + offset
    
    Parameters
    ----------
    time : numpy.ndarray, shape (n,)
        The time values over which the simulated impedance-adjacent experiment takes place
    U : numpy.ndarray with shape (m,)
        Coefficients for the exponential decay functions such that U[i] is the coefficient of 
        exp(-t/tau[i])
    tau : numpy.ndarray with shape (m,)
        The relaxation times used in the model as above.
    offset : float (optional)
        The offset of the steady state signal from 0
    
    Returns 
    -------
    predicted_y : numpy.ndarray with shape (n,)
        predict_y at all time values in t
    """
    # Calculate predict_y(t) for all values in t
    predicted_y = (U @ np.exp(-np.outer(1/tau, time))) + offset

    return predicted_y

def calculate_tau(time):
    """
    Generate a range of tau values for DRT fitting based off an observation window
    over time values 'time'
    
    Parameters
    ----------
    time : arraylike, shape (n,)
        Time values data is observed over

    Returns
    -------
    tau_values : arraylike, shape (m,)
    """
    # Find the smallest time interval in the array
    d_time = [time[i+1] - time[i] for i in range(len(time)-1)]
    d_time_min = np.min(d_time)
    d_T = time[-1]-time[0]

    # Set max and min tau
    tau_min = d_time_min/np.pi
    tau_max = d_T/(2*np.pi)

    # If the number of points in the time array is less than 150,
    # set m = n, else set M = 150
    m = len(time) if len(time) < 150 else 150

    tau_values = np.geomspace(tau_min, tau_max, m)

    return tau_values

def R2_error(y, y_model):
    """
    Calculates the R^2 error between data (y) and model (y_model)

    Parameters
    ----------
    y : arraylike (n,)
        Data used in fit
    y_model : arraylike (n,)
        Model predicted values
    
    Returns
    -------
    R^2 : float 
    """
    SSres = np.mean((y - y_model)**2)
    SStot = np.mean((y - np.mean(y))**2)
    return 1 - SSres/SStot

def mean_square_error(y, y_model):
    """
    Calculates the mean square error (MSE) between data (y) and model (y_model)

    Parameters
    ----------
    y : arraylike (n,)
        Data used in fit
    y_model : arraylike (n,)
        Model predicted values

    Returns
    -------
    MSE : float
    """
    return np.mean((y-y_model)**2)


# Fitting Functions #####################################
def linear_fit(time, y, tau='Auto', offset='Auto', bounds=None, scaling=True):
    """
    Performs a linear fit of 
        y_model = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2)... + U_m*exp(-t/tau_m) + offset
    to the data (y)

    Converts the linear fit problem to a convex quadratic program and minimises using 
    the Operator Splitting Quadratic Program (OSQP) package solver [2]. Using a QP solver 
    was inspired by ADD REFERENCE
    
    Parameters
    ----------
    time : numpy.ndarray, shape (n,)
        Time values over which the simulated experiment took place
    y : numpy.ndarray, shape (n,)
        Data for model fitting
    tau: {'Auto', tuple(tau_min, tau_max, m), array_like of shape (m,)} (optional)
        The tau values used in the model 
            y_model = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2)... + U_m*exp(-t/tau_m) + offset
        'Auto' will use the 'calculate_tau' method to determine lifetimes for fitting. 
        Passing a tuple in the form (tau_min, tau_max, m) will generate a logarithmic array of m values between 
        tau_min and tau_max. 
        Any passed array_like that isn't a tuple of length 3 will be used for all tau values. 
        Default 'Auto'.
    offset : REMOVE THIS
    bounds : {None, tuple(lower, upper)} (optional)
        Sets the bounds for the fitted coeffs. U. Use np.inf for unbounded limit. Default None.
    scaling : bool (optionl)
        Determines whether minmax scaling is applied to the data before fitting 
        (the data is scaled back post fit). Would recomend leaving at default value. 
        Default True.

    Results
    -------
    fit : DRT_Fit_Result
        Fit object containing fit data.
    """
    # Record offset from y=0
    offset = y[-1] if offset == 'Auto' else offset

    # Determine tau 
    if isinstance(tau, str): # Check if tau is set to 'Auto' by verifying it's a string
        tau = calculate_tau(time)
    elif isinstance(tau, tuple) and len(tau) == 3:
        tau = np.geomspace(tau[0], tau[1], tau[3])
    else:
        tau = tau

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
    fit.set_norm_U()
    
    return fit

def checkerboard_fit(time, y, tau='Auto', offset='Auto', checkerboard_iters=50, fit_scaling=True):
    """
    Performs a 'checkerboard' DRT fit by iteratively fitting capacitive and inductive effects in supplied data (y)
    
    time : numpy.ndarray, shape (n,)
        Time values over which the simulated experiment took place
    y : numpy.ndarray, shape (n,)
        Data for model fitting
    tau: {'Auto', tuple(tau_min, tau_max, m), array_like of shape (m,)} (optional)
        The tau values used in the model 
            y_model = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2)... + U_m*exp(-t/tau_m) + offset
        'Auto' will use the 'calculate_tau' method to determine lifetimes for fitting. 
        Passing a tuple in the form (tau_min, tau_max, m) will generate a logarithmic array of m values between 
        tau_min and tau_max. 
        Any passed array_like that isn't a tuple of length 3 will be used for all tau values. 
        Default 'Auto'.
    offset:  {'Auto', float} (optional)
        REMOVE THIS
    checkerboard_iters : int (optional)
        The number of iterations to perform using the checkerboard fitting method
    fit_scaling : bool
        Determines whether minmax scaling is used during each call of the 'linear_fit' method. 
        Can impact the smoothness of the MSE and R^2 over many iterations. Default True.

    Returns
    -------
    fits : arraylike(DRT_Fit_Results), shape (checkerboard_iters,)
        An array of fit objects corresponding to each iteration of the checkerboard procedure.
    """

    # Determine tau
    if isinstance(tau, str):
        tau = calculate_tau(time)
    elif isinstance(tau, tuple) and len(tau) == 3:
        tau = np.geomspace(tau[0], tau[1], tau[3])
    else:
        tau = tau
    
    # Step 1: Prep the signal for fitting by removing offset and scaling
    offset = y[-1] if offset == 'Auto' else offset
    y_max = y.max()
    y_min = y.min()
    scale_factor = y_max-y_min
    y_scaled = (y - y_min)/scale_factor
    y_scaled = y_scaled - y_scaled[-1]

    # Step 2: Set initial values
    U_values = np.zeros(len(tau))
    
    y_cap = y_scaled

    cap_U_values = np.zeros(len(tau))
    ind_U_values = np.zeros(len(tau))

    cap_offset = 0
    ind_offset = 0

    fits = []
    
    # NOTE: we probably want to set the offset and scale factor guesses ourselves
    for i in range(checkerboard_iters):
        
        # Set scale params for capacitive effects and fit
        cap_fit = linear_fit(time, y_cap, tau=tau, offset=cap_offset, scaling=fit_scaling, bounds=(0, np.inf))
        
        # Add the fit to U_values
        cap_U_values = cap_fit.U
        
        # Remove the capacitive effects from y
        y_ind = y_scaled - cap_fit.y
        ind_offset = y_ind[-1]
        
        # Set the inductive scale params and fit by doing a capacitive fit on an inverted function
        ind_fit = linear_fit(time, -y_ind, tau=tau, offset=ind_offset, scaling=fit_scaling, bounds=(0, np.inf))
        
        # Add the fit to U_values 
        ind_U_values = -ind_fit.U

        # Remove the inductive effects from the curve for the next iteration
        y_cap = y_scaled + ind_fit.y
        cap_offset = y_cap[-1]
        
        # Once the fit is done, return a fit object with the results and a dummy cost of 0
        U_values = cap_U_values + ind_U_values

        # Rescale and package results
        U_values = scale_factor*U_values
        y_model = predict_y(time, U_values, tau, offset=offset)
        MSE = mean_square_error(y, y_model)
        R2 = R2_error(y, y_model)

        fit = DRT_Fit_Result(U_values, tau, offset, len(tau), MSE=MSE, R2=R2)
        fit.y = y_model
        fit.set_norm_U()
        fits.append(fit)
        
    return fits

######### Plotting Functions #######################################################################
def plot_y(time, y_model=None, y=None, xaxis_label='Time [s]', yaxis_label='y(t)', 
           y_plot_label='Data', y_model_plot_label='Model', plot_title='DRT Fit', return_ax=False):
    """
    Plot the fitted model agains the data
    
    Parameters
    ----------
    time : numpy.ndarray, shape (n,)
        The time values over which the simulated experiment took place
    y_model : {None, numpy.ndarray shape (n,)} (optional)
        The model predicted curve. Either y_model or y must not be None. y_model is None by default.
    y : {None, numpy.ndarray shape (n,)} (optional)
        The data the model was fitted to. Either y_model or y must not be None. y is None by default.
    xaxis_label : str (optional)
        Label for the x axis of the output plot. 'Time [s]' by default.
    yaxis_label : str (optional)
        Label for the y axis of the output plot. 'y(t)' by default.
    y_plot_label : str (optional)
        Label for the curve y in the output plot legend. 'Data' by default.
    y_model_plot_label : str (optional)
        Label for the y_model curve in the output plot legend. 'Model' by default.
    plot_title : str (optional)
        Title of the output plot. 'DRT Fit' by default.
    return_ax : bool (optional)
        Determines whether the matplotlib.axes.Axes object is returned (True) or plotted (False). 
        Default False.

    Returns
    -------
    ax : matplotlib.axes.Axes (optional)
        Returned axes object if return_ax is True.
    """
    
    # Set up the plot
    fig, ax = plt.subplots()
    if y_model is None and y is None:
        raise ValueError("At least one of 'y_model' and 'y' must not be 'None'")
    
    if y is not None:
        ax.plot(time, y, label=y_plot_label)

    if y_model is not None:
        ax.plot(time, y_model, label=y_model_plot_label)

    ax.set_xscale('log')
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)
    ax.set_title(plot_title)
    ax.legend()
    
    # Return axis if called for, otherwise show the figure
    if return_ax:
        return ax 
    else: 
        plt.show()


def plot_U(tau_model, U_model, tau_analytic=None, U_analytic=None, xaxis_label='$\\tau$ [s]', 
           yaxis_label='U', U_model_label='Model', U_label='Analytic', plot_title='DRT', return_ax=False):
    """
    Plot the fitted DRT against an optional analytic DRT
    
    Parameters
    ----------
    tau_model : numpy.ndarray, shape (m,)
        The tau values used in the model
    U_model : numpy.ndarray shape (m,)
        The coeffs from the DRT fit, given by U in 
            y_model = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2)... + U_m*exp(-t/tau_m) + offset
    tau_analytic : {None, numpy.ndarray shape (l,)} (optional)
        Array of tau values use to generate the analytic curve, if known. This is mostly for 
        comparision if fitting to a known DRT for testing purposes. Default None.
    U_analytic : {None, numpy.ndarray shape (l,)} (optional)
        Array of U values used in the analytic curve, if known. This is mostly for 
        comparision if fitting to a known DRT for testing purposes. Default None.
    xaxis_label : str (optional)
        Label for the x axis of the output plot. '$\\tau$ [s]' by default.
    yaxis_label : str (optional)
        Label for the y axis of the output plot. 'U' by default.
    U_model_label : str (optional)
        Label for the curve U_model in the output plot legend. 'Model' by default.
    U_label : str (optional)
        Label for the U_analytic curve in the output plot legend. 'Analytic' by default.
    plot_title : str (optional)
        Title of the output plot. 'DRT' by default.
    return_ax : bool (optional)
        Determines whether the matplotlib.axes.Axes object is returned (True) or plotted (False). 
        Default False.

    Returns
    -------
    ax : matplotlib.axes.Axes (optional)
        Returned axes object if return_ax is True.
    """
    # Set up the plot
    fig, ax = plt.subplots()
    if tau_analytic is not None and U_analytic is not None:
        ax.plot(tau_analytic, U_analytic, label=U_label)
    ax.plot(tau_model, U_model, label=U_model_label)
    ax.set_xscale('log')
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)
    ax.set_title(plot_title)
    ax.legend()

    # Return axis if called for, otherwise show the figure
    if return_ax:
        return ax 
    else: 
        plt.show()

def plot_cumulative_U(tau_model, U_model, tau_analytic=None, U_analytic=None, xaxis_label='$\\tau$ [s]', 
                      yaxis_label='Cumulative U', U_model_label='Model', U_label='Analytic', 
                      plot_title='Cumulative DRT', return_ax=False):
    """
    Plot the fitted DRT against an optional analytic DRT
    
    Parameters
    ----------
    tau_model : numpy.ndarray, shape (m,)
        The tau values used in the model
    U_model : numpy.ndarray shape (m,)
        The coeffs from the DRT fit, given by U in 
            y_model = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2)... + U_m*exp(-t/tau_m) + offset
    tau_analytic : {None, numpy.ndarray shape (l,)} (optional)
        Array of tau values use to generate the analytic curve, if known. This is mostly for 
        comparision if fitting to a known DRT for testing purposes. Default None.
    U_analytic : {None, numpy.ndarray shape (l,)} (optional)
        Array of U values used in the analytic curve, if known. This is mostly for 
        comparision if fitting to a known DRT for testing purposes. Default None.
    xaxis_label : str (optional)
        Label for the x axis of the output plot. '$\\tau$ [s]' by default.
    yaxis_label : str (optional)
        Label for the y axis of the output plot. 'Cumulative U' by default.
    U_model_label : str (optional)
        Label for the curve U_model in the output plot legend. 'Model' by default.
    U_label : str (optional)
        Label for the U_analytic curve in the output plot legend. 'Analytic' by default.
    plot_title : str (optional)
        Title of the output plot. 'Cumulative DRT' by default.
    return_ax : bool (optional)
        Determines whether the matplotlib.axes.Axes object is returned (True) or plotted (False). 
        Default False.

    Returns
    -------
    ax : matplotlib.axes.Axes (optional)
        Returned axes object if return_ax is True.
    """

    # Calculate the cumulative U values
    cumulative_U_model = [np.sum(U_model[:i]) for i in range(len(U_model))]
    if tau_analytic is not None and U_analytic is not None:
        cumulative_U_analytic = [np.sum(U_analytic[:i]) for i in range(len(U_analytic))]
    
    # Set up the plot
    fig, ax = plt.subplots()
    if tau_analytic is not None and U_analytic is not None:
        ax.plot(tau_analytic, cumulative_U_analytic, label=U_label)
    ax.plot(tau_model, cumulative_U_model, label=U_model_label)
    ax.set_xscale('log')
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)
    ax.set_title(plot_title)
    ax.legend()

    # Return axis if called for, otherwise show the figure
    if return_ax:
        return ax 
    else: 
        plt.show()

def plot_MSE(error_array, xaxis_label='Iteration', yaxis_label='MSE', plot_title='MSE per Fit Iteration', 
             return_ax=False):
    """
    Plots the mean square error. Useful for checking how the MSE varies with iteration 
    during checkerboard fit.
    
    Parameters
    ----------
    error_array : {array_like(DRT_Fit_Result) of shape (i,), arraylike(float)}
        An iterable conatining EITHER DRT_Fit_Result objects from which the MSE values are taken 
        OR float values which are considered to be the MSE values
    xaxis_label : str (optional)
        Label for the x axis of the output plot. 'Iteration' by default.
    yaxis_label : str (optional)
        Label for the y axis of the ouput plot. 'MSE' by default.
    plot_title : str (optional)
        Title for the output plot. 'MSE per Fit Iteration' by default.
    return_ax : bool (optional)
        Determines whether the matplotlib.axes.Axes object is returned (True) or plotted (False). 
        Default False.
        
    Returns
    -------
    ax : matplotlib.axes.Axes (optional)
        Returned axes object if return_ax is True.
    """
    
    if isinstance(error_array[0], DRT_Fit_Result):
        MSE_values = [fit.MSE for fit in error_array]
    else: 
        MSE_values = error_array

    fig, ax = plt.subplots()
    ax.plot(range(1, len(error_array)+1), MSE_values)
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)
    ax.set_title(plot_title)

    if return_ax:
        return ax
    else:
        plt.show()

def plot_R2(error_array, ylim=(0, 1.1), xaxis_label='Iteration', yaxis_label='$R^2$', plot_title='$R^2$ per fit Iteration', 
             return_ax=False):
    """
    Plots the R^2 error. Useful for checking how R^2 varies with iteration 
    during checkerboard fit.
    
    Parameters
    ----------
    fit_array : array_like(DRT_Fit_Result), shape (i,)
        An iterable conatining DRT_Fit_Result objects from which the R^2 values are taken
    ylim : tuple, (2,) (optional)
        Set the ylimits of the plot. ylim[0] gives the lower bound, y[1] the upper. Default (0, 1.1)
    xaxis_label : str (optional)
        Label for the x axis of the output plot. 'Iteration' by default.
    yaxis_label : str (optional)
        Label for the y axis of the ouput plot. '$R^2$' by default.
    plot_title : str (optional)
        Title for the output plot. '$R^2$ per fit Iteration'.
    return_ax : bool (optional)
        Determines whether the matplotlib.axes.Axes object is returned (True) or plotted (False). 
        Default False.
        
    Returns
    -------
    ax : matplotlib.axes.Axes (optional)
        Returned axes object if return_ax is True.
    """
    
    if isinstance(error_array[0], DRT_Fit_Result):
        R2_values = [fit.R2 for fit in error_array]
    else:
        R2_values = error_array

    fig, ax = plt.subplots()
    ax.plot(range(1, len(error_array)+1), R2_values)
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel(yaxis_label)
    ax.set_ylim(ylim)
    ax.set_title(plot_title)

    if return_ax:
        return ax
    else:
        plt.show()

######### Data Saving and Reading ####################################################################
def saveModelsToTxt(path, fits, float_format='%.5e'):
    """
    Save the tau and U values from a collection of DRT_Fit_Result objects to a txt file.

    Parameters
    ----------
    path : str
        File path of save file
    fits : arraylike(DRT_Fit_Result)
        Array like object of fit results 
    float_format: str (optional)
        Controls how float values are save to file

    Returns
    -------
    None
    """
    tau = fits[0].tau # All iterations in a checkerboard fit will have the same tau
    offset = fits[0].offset # All iterations in a checkerboard fit will have the same offset
    DRT_data = {'tau' : tau}
    for i in range(len(fits)):
        DRT_data[f'U_iter_{i+1}'] = fits[i].U
    DRT_data = pd.DataFrame(DRT_data)
    DRT_data.to_csv(path, sep=' ', float_format=float_format, index=False)

def saveModelPredictionsToTxt(path, fits, time, float_format='%.5e'):
    """
    Save the y values from a collection of DRT_Fit_Result objects to a txt file.

    Parameters
    ----------
    path : str
        File path of save file
    fits : arraylike(DRT_Fit_Result)
        Array like object of fit results 
    time : numpy.ndarray
        Time values over which each fit was made
    float_format: str (optional)
        Controls how float values are save to file

    Returns
    -------
    None
    """
    model_predictions = {'t' : time} ###FIX THIS 
    for i in range(len(fits)):
        model_predictions[f'y_model_iter_{i+1}'] = fits[i].y
    model_predictions = pd.DataFrame(model_predictions)
    model_predictions.to_csv(path, sep=' ', float_format=float_format, index=False)

def saveModelErrorsToTxt(path, fits, float_format='%.5e'):
    """
    Save the MSE and R2 values from a collection of DRT_Fit_Result objects to a txt file.

    Parameters
    ----------
    path : str
        File path of save file
    fits : arraylike(DRT_Fit_Result)
        Array like object of fit results 
    float_format: str (optional)
        Controls how float values are save to file

    Returns
    -------
    None
    """
    MSE = [fit.MSE for fit in fits]
    R2 = [fit.R2 for fit in fits]
    model_errors = pd.DataFrame({'MSE' : MSE, "R2" : R2})
    model_errors.to_csv(path, sep=' ', float_format=float_format, index=False)

def saveToTxt(directory_path, fits, time, float_format='%.5e'):
    """
    Save tau, U, y, MSE, and R2 values from a collection of fit objects to text files.

    Parameters
    ----------
    directory_path : str
        All save files are saved to this directory
    fits : arraylike(DRT_Fit_Result)
        Array like object of fit results 
    time : numpy.ndarray
        Time values over which each fit was made
    float_format: str (optional)
        Controls how float values are save to file

    Returns
    -------
    None
    """
    # Check the directory exists and make it if not
    try: 
        os.makedirs(directory_path)
    except FileExistsError:
        pass
    
    # Define file names
    DRTModels_filename = directory_path + "/DRTModels.txt"
    modelPredictions_filename = directory_path + "/modelOutputs.txt"
    outputErrors_filename = directory_path + "/outputErrors.txt"
    
    # Save data
    saveModelsToTxt(DRTModels_filename, fits, float_format=float_format)
    saveModelPredictionsToTxt(modelPredictions_filename, fits, time, float_format=float_format)
    saveModelErrorsToTxt(outputErrors_filename, fits, float_format=float_format)

def readFromTxt(path):
    """
    Reads data stored in a space seperated text file

    Parameters
    ----------
    path : str
        Path of file

    Returns
    -------
    None
    """
    return pd.read_csv(path, sep=r"\s+")

def saveToPickle(path, fits):
    """
    Saves an array of DRT_Fit_Results to a .pkl file

    Parameters
    ----------
    path : str
        Path of file
    fits : arraylike(DRT_Fit_Result)
        Array like object of fit results 

    Returns
    -------
    None
    """
    with open(path, 'wb') as file:
        pickle.dump(fits, file)

def readFromPickle(path):
    """
    Loads an array of DRT_Fit_Results from a .pkl file

    Parameters
    ----------
    path : str
        Path of file

    Returns
    -------
    fits : arraylike(DRT_Fit_Result)
        Array of DRT_Fit_Result objects read from the .pkl save file
    """
    with open(path, 'rb') as file:
        fits = pickle.load(file)
    return fits

######### Scripting Functionality ####################################################################
if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("dataFile", 
                        help="path to data file containing time and function values for fit")
    parser.add_argument("-DRTDirectory", default='DRT_Saves', 
                        help="path to directory for DRT save files (default: DRT_Saves)")
    parser.add_argument("-timeCol", default= 't', 
                        help="heading of the time column in dataFile (default: 't')")
    parser.add_argument("-funcCol", default = 'Jext', 
                        help="heading of the function column in dataFile (default: 'Jext')")
    parser.add_argument("-iters", type=int, default=50, 
                        help="number of iterations in checkerboard fit (default: 50)")
    parser.add_argument("-saveFormat", default="txt", choices=["txt", "pkl"],
                        help="saved data file format (default: txt)")
    args = parser.parse_args()

    # Read data 
    try:
        dataFile = pd.read_csv(args.dataFile, sep=r"\s+")
    except FileNotFoundError:
        print(f"Error: dataFile not found")
        sys.exit(1)
  
    try: 
        time = np.array(dataFile[args.timeCol])
    except KeyError:
        print(f"Error: column '{args.timeCol}' not found in '{args.dataFile}'")
        sys.exit(1)

    try: 
        y = np.array(dataFile[args.funcCol])
    except KeyError:
        print(f"Error: column '{args.funcCol}' not found in '{args.dataFile}'")
        sys.exit(1)

    # Check whether DRTDirectory exists and, if not, create it
    try: 
        os.makedirs(args.DRTDirectory)
    except FileExistsError:
        pass
    except PermissionError:
        print(f"Error: DRT.py lacks the neccessary permissions to create {args.DRTDirectory}. Try manually creating {args.DRTDirectory} instead.")
        sys.exit(1)
    except Exception as error:
        print(f"Error: {error}")
        sys.exit(1)
    
    # Run checkerboard fit 
    run_code = 0
    try:
        fits = checkerboard_fit(time, y, tau='Auto', offset='Auto', checkerboard_iters=args.iters)
    except Exception as error:
        traceback.print_exc()
        sys.exit(1)
    
    # Save DRT data to file
    if args.saveFormat == "txt":
        saveToTxt(args.DRTDirectory, fits, time)
    elif args.saveFormat == "pkl":
        saveToPickle(args.DRTDirectory + "/models.pkl", fits)

    sys.exit(0)