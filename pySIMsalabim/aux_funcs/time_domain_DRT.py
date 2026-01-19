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
    def __init__(self, U, tau, offset, cost, m):
        self.U = U
        self.tau = tau 
        self.offset = offset 
        self.cost = cost
        self.m = m 
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
        self.y = predict_y(t, self.U, self.tau, self.offset, backend=backend, device=device)


class DRTLinearModel(torch.nn.Module):
    def __init__(self, tau, U_scale_factor=1, offset=0, bounds=None, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.tau = torch.tensor(tau, device=self.device, dtype=torch.float32)
        self.m = len(tau)
        
        # Define the parameters object, which will hold U and tau
        self.params = torch.nn.Parameter(torch.concat((torch.ones(self.m, device=device, dtype=torch.float32)/self.m*U_scale_factor, 
                                                       torch.tensor([offset], device=device, dtype=torch.float32))))
        
        # Set bounds for U values if given 
        if bounds is not None:
            self.lower_bound = bounds[0]
            self.upper_bound = bounds[1]
        else:
            self.lower_bound = None
            self.upper_bound = None
        
    def forward(self, t):
        if self.lower_bound is not None:
            params_clamped = torch.clamp(self.params, self.lower_bound, self.upper_bound)
        else:
            params_clamped = self.params
        return predict_y_torch(t, params_clamped[:-1], self.tau, params_clamped[-1])


######### Function Definitions ####################################################################

# Utility Functions #####################################
def numpy_converter(x):
    """Makes sure arraylike object is a numpy ndarray

    Parameters
    ----------
    x : arraylike 
    
    Returns
    -------
    x : np.ndarray

    """
    if isinstance(x, np.ndarray):
        x = x
    elif isinstance(x, list):
        x = np.array(x)
    elif isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        raise TypeError("Argument is not of type np.array, list, or torch.Tensor")
    return x
    
def torch_tensor_converter(x, device=torch.device('cpu')):
    """Makes sure arraylike object is a torch.Tensor

    Parameters
    ----------
    x : arraylike 
    
    Returns
    -------
    x : torch.Tensor on specified devicee

    """
    # Check if the device being used is an accelerator, and downcast floats if so
    if isinstance(x, torch.Tensor):
        x = x.to(torch.float32).to(device)
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(torch.float32).to(device)
    elif isinstance(x, list):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        raise TypeError("Argument is not of type np.array, list, or torch.Tensor")
    return x

# Regularisation Functions ##############################
    
def IC_ratio_reg(U, alpha=0.01, backend='numpy', device=torch.device('cpu')):
    # Regularisation parameter inspired by the mu criterion in [1]
    if backend == 'numpy':
        U = numpy_converter(U)
        return alpha*(np.absolute(U[np.where(U < 0)]).sum() / np.absolute(U[np.where(U >= 0)]).sum())
    elif backend == 'torch':
        U = torch_tensor_converter(U, device=device)
        return alpha*(torch.absolute(U[torch.where(U < 0)]).sum() / torch.absolute(U[torch.where(U >= 0)]).sum())

def ridge_reg(x, alpha=0.01):
    x = numpy_converter(x)
    return alpha*np.square(x)

def lasso_reg(x, alpha=0.01):
    x = numpy_converter(x)
    return 

# Fitting Functions #####################################

def predict_y_numpy(t, U, tau, offset=0):
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


def predict_y_torch(t, U, tau, offset=0):
    """Use torch.Tensor objects to calculate the function
        predict_y(t) = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2) + ... + U_m*exp(-t/tau_m) + offset
    
    Parameters
    ----------
    t : torch.Tensor, shape (n,)
        The time values over which the simulated impedance-adjacent experiment takes place
    U : torch.Tensor, shape (m,)
        Coefficients for the exponential decay functions such that U[i] is the coefficient of 
        exp(-t/tau[i])
    tau : torch.Tensor with shape (m,)
        The distribution of relaxation times
    offset : float (optional)
        Initial guess of the predict_y offset value
    
    Returns 
    -------
    predicted_y : torch.Tensor with shape (n,)
        predict_y at all time values in t
    """

    # Calculate predict_y(t) for all values in t
    predicted_y = (U @ torch.exp(-torch.outer(1/tau, t))) + offset
    
    return predicted_y


def predict_y(t, U, tau, offset=0, backend='numpy', device=torch.device('cpu')):
    """Calculate the function
        predict_y(t) = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2) + ... + U_m*exp(-t/tau_m) + offset
    
    Parameters
    ----------
    t : array-like, shape (n,)
        The time values over which the simulated impedance-adjacent experiment takes place
    U : array-like, shape (m,)
        Coefficients for the exponential decay functions such that U[i] is the coefficient of 
        exp(-t/tau[i])
    tau : array-like with shape (m,)
        The distribution of relaxation times
    offset : float (optional)
        Initial guess of the predict_y offset value
    backend : {'numpy', 'torch'} (optional)
        Determines whether matrix operations are done with numpy or pytorch. Default numpy. 
    device : torch.device (optional)
        Determines what device is used for matrix ops if backend='torch'.
    
    Returns 
    -------
    predicted_y : numpy.ndarray or torch.Tensor with shape (n,)
        predict_y at all time values in t
    """
    
    if backend == 'numpy':
        t = numpy_converter(t)
        U = numpy_converter(U)
        tau = numpy_converter(tau)

        return predict_y_numpy(t, U, tau, offset=offset)

    elif backend == 'torch':
        t = torch_tensor_converter(t, device=device)
        U = torch_tensor_converter(U, device=device)
        tau = torch_tensor_converter(tau, device=device)

        return predict_y_torch(t, U, tau, offset)


def groningen_fit(t, y, U0, tau0, offset0=0, predict_y=True, **kwargs):
    """Fits a DRT to a function y(t) using a non-linear approach

    Uses a non-linear approach to fit 
        predict_y(t) = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2) + ... + U_m*exp(-t/tau_m) + offset
    to a function y by finding the optimal values for the parameters 
        U = [U_1, U_2, ..., U_m] 
        tau = [tau_1, tau_2, ..., tau_m]
        
    Parameters
    ----------
    t : (list/numpy.ndarray) with shape (n,)
        The time values over which the function y is known
    y : list/numpy.ndarray with shape (n,)
        The values of y(t) that predict_y will be fitted to
    U0 : list/numpy.ndarray with shape (m,)
        Initial guess of U values for fitting 
    tau0 : list/numpy.ndarray with shape (m,)
        Initial guess of tau values for fitting
    offset0 : float (optional)
        Initial guess of the predict_y offset value
    predict_y : bool (optional)
        If true, will set the predict_y attribute of the returned DRT_Fit_Result object using t
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
                Number of lifetimes/coefficients used in predict_y, given by the length of U/tau
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
    error = lambda x : y - predict_y(t, x[:m], x[m:-1], offset=x[-1])
    
    # Fit the params using scipy. Note x_scale='jac' makes for better fits than the default value
    fit = so.least_squares(error, x0=initial_params, x_scale='jac', **kwargs)
    
    # Store the fit results
    drt_fit = DRT_Fit_Result(fit.x[:m], fit.x[m:-1], fit.x[-1], fit.cost, m)

    # Set the predict_y curves for each DRT_Fit_Result result if calc_predict_y=True
    drt_fit.predict_y(t) if predict_y else None
    
    return drt_fit


def multi_groningen_fit(t, y, m_values, U_scale_factor=0, offset0=0, predict_y=True, **kwargs):
    """ Runs groningen_fit over multiple m values 
    
    Parameters
    ----------
    t : list or numpy.ndarray, shape (n,)
        Time values over which y(t) is known
    y : list or numpy.ndarray, shape(n, )
        y(t) values each predict_y curve is fitted to
    m_values : list, shape (i,)
        Various m values for groningen_fit, where m is the number of coefficients/lifetimes
        in predict_y(t)
    U_scale_factor : float
        Controls the scale for the intial guess of the coefficients U0 passed to groningen_fit
    offset0 : float 
        Initial guess for the offset passed to groningen_fit
    predict_y : bool (optional)
        If true, will set the predict_y attribute of the returned DRT_Fit_Result objects using t
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
        U0 = np.ones(m)*U_scale_factor/m
        tau0 = np.geomspace(t[0], t[-1], m)
        offset0 = offset0
        
        # Fit the curve
        fit = groningen_fit(t, y, U0, tau0, offset0=offset0, predict_y=predict_y, **kwargs)
        
        # Add to fits
        fits.append(fit)
        
    return fits


def least_squares_linear_fit(t, y, tau, U_scale_factor=1, offset=0, alpha=0, predict_y=True, backend='numpy', device='cpu', **kwargs):
    """Fits an predict_y(t) curve to a function y(t) using a grid-based approach

        Uses a linear approach to fit 
            predict_y(t) = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2) + ... + U_m*exp(-t/tau_m) + offset
        to a function y by finding the optimal values for the parameters 
            U = [U_1, U_2, ..., U_m] 
            tau = [tau_1, tau_2, ..., tau_m]
            
        Parameters
        ----------
        t : (list/numpy.ndarray) with shape (n,)
            The time values over which the function y is known
        y : list/numpy.ndarray with shape (n,)
            The values of y(t) that predict_y will be fitted to
        tau : list/numpy.ndarray with shape (m,)
            Grid of tau values used in calculating predict_y(t)
        U_scale_factor : float
            Determines the magnitude and polarity of initial guess for U
        offset0 : float (optional)
            Initial guess of the predict_y offset value
        predict_y : bool (optional)
            If true, will set the predict_y attribute of the returned DRT_Fit_Result object using t
        backend : {'numpy', 'torch'} (optional)
            Determines whether matrix operations are done with numpy or pytorch. Default numpy. 
        device : {'cpu', 'acc', torch.device} (optional)
            Determines what device is used for matrix ops if backend='torch'. 'cpu' runs all calculations
            on the CPU and 'acc' will run calculations on a accelerator (cuda, mps, etc) if available.
            Any passed torch.device object will be used.
        kwargs
            Keyword arguments passed to scipy.optimize.least_squares for fitting
        
        Returns
        -------
        drt_fit : DRT_Fit_Result 
            With attributes as follows
                U : numpy.ndarray or torch.Tensor, shape (m,)
                    Fitted [U_1, U_2, ..., U_m] array
                tau : numpy.ndarray or torch.Tensor, shape (m, )
                    Fitted [tau_1, tau_2, ..., tau_m] array
                offset : numpy.float64
                    Fitted value for offset
                cost : numpy.float64
                    The value of the cost function 
                        F(x) = 0.5 * sum(rho(error_i(x)**2), i = 0, ..., n - 1)
                    at the solution. By default, rho(z) = z. See scipy.optimize.least_squares for details
                m : int
                    Number of lifetimes/coefficients used in predict_y, given by the length of U/tau
        """
    # Create the error function 
    if backend == 'numpy':
        # Make sure passed parameters are numpy.ndarrays
        t = numpy_converter(t)
        y = numpy_converter(y)
        tau = numpy_converter(tau)

        # Create initial guess for U 
        m = len(tau)
        U0 = U_scale_factor*np.ones(m)/m

        # Create an initial array-like guess of variables
        x0 = np.concatenate((U0, [offset]))

        # Define the error function
        error = lambda x : y - predict_y(t, x[:-1], tau, offset=x[-1], backend='numpy')
        
        # Minimise the error function using scipy
        fit = so.least_squares(error, x0=x0, **kwargs)

        # Store the results in a DRT_Fit object
        drt_fit = DRT_Fit_Result(fit.x[:-1], tau, fit.x[-1], fit.cost, m)

        # Set the predict_y curve for DRT_Fit_Result result if calc_predict_y=True
        drt_fit.predict_y(t) if predict_y else None

        # Assume params is of dimension 2 x m, where m is the number of tau/U_values
        return drt_fit

    elif backend == 'torch':
        # Check which device should be used for calculations 
        if device == 'cpu': 
            device = torch.device('cpu')
        elif device == 'acc':
            if torch.accelerator.is_available(): 
                device = torch.accelerator.current_accelerator() 
            else:
                raise Exception("Accelerator unavailable")
        elif isinstance(device, torch.device):
            device = device
        else:
            raise Exception("Device needs to be 'cpu', 'acc', or of type torch.device")

        # Make sure passed parameters are torch.Tensor objects
        t = torch_tensor_converter(t, device=device)
        y = torch_tensor_converter(y, device=device)
        tau = torch_tensor_converter(tau, device=device)
        
        # Create initial guess for U 
        m = tau.size(dim=0)
        U0 = U_scale_factor*torch.ones(m, device=device)/m

        # Create an initial array-like guess of variables
        x0 = torch.cat((U0, torch.tensor([offset], device=device, dtype=torch.float32)))

        error = lambda x : y - predict_y(t, x[:-1], tau, offset=x[-1], backend='torch', device=device)

        # Minimise the error function using scipy
        fit = so.least_squares(error, x0=x0, **kwargs)

        # Store the results in a DRT_Fit object
        drt_fit = DRT_Fit_Result(fit.x[:-1], tau, fit.x[-1], fit.cost, m)

        # Set the predict_y curve for DRT_Fit_Result result if calc_predict_y=True
        drt_fit.predict_y(t) if predict_y else None

        # Assume params is of dimension 2 x m, where m is the number of tau/U_values
        return drt_fit
        
        
def alpha_linear_fit(t, y, tau, U_scale_factor='Auto', offset='Auto', predict_y=True, alpha=0, max_step_iter=20, max_iter=200, device='cpu', bounds=None, **kwargs):
    """Fits a distribution of relaxation times (tau) to a decay process (y) using gradient descent 

    Parameters
    ----------
        t : arraylike, shape (n,)
            Time values for y(t)
        y : arraylike, shape (n,)
            Function y(t) for fitting
        tau : arraylike, shape (m,)
            Relaxation times used in the fit
        U_scale_factor : {'Auto' or float} (optional)
            Scaling parameter for initial U guess
        offset : {'Auto' or float'} (optional)
            Initial fit offset guess
        predict_y : bool
            If true, will set the predict_y attribute of the returned DRT_Fit_Result object using t
        alpha : int or float
            mu regularisation strength in alpha_linear_fit. Default 0 means there is no regularisation
        max_step_iter : int (optional)
            Maximum number of optimizer iterations per gradient descent step
        max_iter : int (optional)
            Maximum number of total optimizer iterations
        device : {'cpu', 'acc', torch.device} (optional)
            Determines which device is used for torch-based calculations. 'cpu' runs all calculations
            on the CPU and 'acc' will run calculations on a accelerator (cuda, mps, etc) if available.
            Any passed torch.device object will be used.  
        bounds : tuple, shape (2,)
            Bounds for parameters U

    Raises
    ------
        Exception: Accelerator device isn't available
        Exception: Device isn't one of the 3 permissible options

    Returns
    -------
        final_fit_result : DRT_Fit_Result
            Object containing key fit data
    """
    
    
    # Get the device type
    if device == 'cpu': 
            device = torch.device('cpu')
    elif device == 'acc':
        if torch.accelerator.is_available(): 
            device = torch.accelerator.current_accelerator() 
        else:
            raise Exception("Accelerator unavailable")
    elif isinstance(device, torch.device):
        device = device
    else:
        raise Exception("Device needs to be 'cpu', 'acc', or of type torch.device")
    
    # Set U scale factor 
    if U_scale_factor == 'Auto':
        U_scale_factor = np.max(y) - np.min(y)
    else: 
        U_scale_factor = U_scale_factor
    
    # Set offset
    if offset == 'Auto':
        offset = np.min(y)
    else: 
        offset = offset
        
    # Convert input into tensors 
    t = torch_tensor_converter(t, device=device)
    y = torch_tensor_converter(y, device=device)
    
    # Construct the pytorch model
    linear_model = DRTLinearModel(tau, U_scale_factor=U_scale_factor, offset=offset, bounds=bounds, device=device)
    
    # Define an optimizer 
    optimizer = torch.optim.LBFGS(linear_model.parameters(), max_iter=max_step_iter, history_size=10)
    
    # Loss History
    loss_history = []
    
    # Define the closure function to be called for each improvement step of the optimizer
    def closure():
        # Set gradients of optimiser with respect to all weights to 0
        optimizer.zero_grad()
        
        # Predict y with the current DRT
        predictions = linear_model(t)
                
        # Calculate the difference between y and the model prediction, including the IC regulariser 
        loss = torch.sum((predictions - y)**2) + alpha*IC_ratio_reg(linear_model.params[:-1], alpha=alpha, backend='torch', device=device)
        
        # Calculate the gradients of the loss function with regards to each parameter
        loss.backward()
        
        # Record the numerical loss
        loss_history.append(loss.item())
        return loss
    
    for epoch in range(max_iter // max_step_iter):
        # Update parameters by taking a 'step' along the inverse of the weight gradient
        optimizer.step(closure)
        
        # Check if bounding is applied to the U parameters, and 'clamp' the values within the bounds if so
        if bounds:
            for param in linear_model.parameters():
                param.data.clamp_(bounds[0], bounds[1])

        if len(loss_history) > 1:
            if abs(loss_history[-1] - loss_history[-2]) < 1e-9:
                break
    
    # Get final loss, and final 
    final_loss = loss_history[-1]
    final_params = linear_model.params.detach().cpu().numpy()
    U = final_params[:-1]
    offset = final_params[-1]
    
    # Bundle results into a DRT_Fit_Result object
    final_fit_result = DRT_Fit_Result(U, tau, offset, final_loss, m=len(tau))
    
    if predict_y:
        final_fit_result.predict_y(t)
    
    return final_fit_result


def checkerboard_fit(t, y, tau, alpha=0, checkerboard_iter=30, max_fit_iter=500, max_step_fit_iter=20, device='cpu', verbose=False, **kwargs):
    """Fits a distribution of relaxation times (tau) to a decay process (t) using the 'checkerboard' method
    
    The checkerboard method operates as follows:
        1. Set y_cap = y
        2. Call alpha_linear_fit on y_cap to fit capacitive lifetimes 
        3. Remove capacitive fit from y to get y_ind
        4. Call alpha_linear_fit on y_ind to fit inductive lifetimes
        5. Remove inductive effects from y to get y_cap
        6. Repeat 2-5 for the specified number of iterations 

    Parameters
    ----------
    t : arraylike, shape (n,)
        Time values for y(t)
    y : arraylike, shape (n,)
        Function y(t) for fitting
    tau : arraylike, shape (m,)
        Relaxation times used in the fit
    alpha : int or float
        mu regularisation strength in alpha_linear_fit. Default 0 means there is no regularisation
    checkerboard_iter : int (optional)
        Number of times the checkerboard iteration algorithm is run
    max_fit_iter : int (optional)
        Maximum iterations for each alpha_linear_fit call
    max_step_fit_iter : int (optional)
        Maximum iterations per step for each alpha_linear_fit call
    device : {'cpu', 'acc', torch.device} (optional)
        Determines which device is used to torch-based calculations. 'cpu' runs all calculations
        on the CPU and 'acc' will run calculations on a accelerator (cuda, mps, etc) if available.
        Any passed torch.device object will be used.    
    kwargs : 
        Key word arguments passed to alpha_linear_fit
    
    Returns
    -------
        fits : list, shape (checkerboard_iter,)
            List of fit objects corresponding to each iteration of the checkerboard fit algorithm 
    """
    offset = y[-1]
    U_values = np.zeros(len(tau))
    
    y_cap = y

    cap_U_values = np.zeros(len(tau))
    ind_U_values = np.zeros(len(tau))

    cap_offset = offset
    ind_offset = 0

    fits = []
    
    MSE = []
    cost = []
    
    # NOTE: we probably want to set the offset and scale factor guesses ourselves
    for i in range(checkerboard_iter):
        
        # Set scale params for capacitive effects and fit
        cap_fit = alpha_linear_fit(t, y_cap, tau, device=device, offset=cap_offset, alpha=alpha, 
                                       max_step_iter=max_step_fit_iter, max_iter=max_fit_iter, bounds=(0, np.inf), **kwargs)
        
        # Add the fit to U_values
        cap_U_values = cap_fit.U
        
        # Remove the capacitive effects from y
        y_ind = y - cap_fit.y
        ind_offset = y_ind[-1]
        
        # Set the inductive scale params and fit by doing a capacitive fit on an inverted function
        ind_fit = alpha_linear_fit(t, -y_ind, tau, device=device, offset=-ind_offset, alpha=alpha, 
                                       max_step_iter=max_step_fit_iter, max_iter=max_fit_iter, bounds=(0, np.inf), **kwargs)
        
        # Add the fit to U_values 
        ind_U_values = -ind_fit.U

        # Remove the inductive effects from the curve for the next iteration
        y_cap = y + ind_fit.y
        cap_offset = y_cap[-1]
        
        # Once the fit is done, return a fit object with the results and a dummy cost of 0
        U_values = cap_U_values + ind_U_values
        fit = DRT_Fit_Result(U_values, tau, offset, 0, len(tau))
        
        # Calculate the cost 
        fit.predict_y(t)
        MSE.append(np.mean((y-fit.y)**2))
        cost.append(np.sum((fit.y - y)**2) + alpha*IC_ratio_reg(fit.U, alpha=alpha, backend='torch', device=device))
        fit.cost = np.mean((y-fit.y)**2)
        fits.append(fit)
        
        if verbose == True:
            plt.plot(t, y, label='Sim')
            plt.plot(t, cap_fit.y, linestyle='-.', label='cap')
            plt.plot(t, -ind_fit.y, linestyle=':', label='ind')
            plt.xscale('log')
            plt.xlabel("$t$ [$\\text{s}$]")
            plt.ylabel("$J$")
            plt.xscale('log')
            plt.title(f"Perfect Impedance Curve {i}")
            plt.legend()
            plt.show()
            
            plt.plot(tau, cap_U_values, label='Cap', linestyle='-.')
            plt.plot(tau, -ind_U_values, label='ind', linestyle=':')
            plt.xscale('log')
            plt.xlabel("$\\tau$ [$\\text{s}$]")
            plt.ylabel("$U$")
            plt.xscale('log')
            plt.title(f"Distribution of Relaxation Times {i}")
            plt.show()
            
            plt.plot(t, y - cap_fit.y, linestyle='-.', label='cap')
            plt.xscale('log')
            plt.xlabel("$t$ [$\\text{s}$]")
            plt.ylabel("Residual $J - \hat{J}_{\\text{cap}}$")
            plt.xscale('log')
            plt.title(f"Residuals post capacitive fit{i}")
            plt.legend()
            plt.show()
    
    if verbose == True:
        plt.plot(range(checkerboard_iter), MSE)
        min_mse_index = np.argmin(MSE)
        plt.axvline(min_mse_index, label=f"index: {min_mse_index}")
        plt.xlabel("Iteration")
        plt.ylabel("MSE")
        plt.legend()
        plt.show()
        
        plt.plot(range(checkerboard_iter), cost)
        min_cost_index = np.argmin(cost)
        plt.axvline(min_cost_index)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()
    
    return fits


def osqp_linear_fit(time, y, tau, offset='Auto', bounds=None):
    """
    Converts the linear fit problem to a convex quadratic program and minimises using 
    the Operator Splitting Quadratic Program (OSQP) package solver [2].
    """
    # Remove offset from signal to prep for quadratic form
    offset = y[-1] if offset == 'Auto' else offset
    y = y - offset
    
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
    
    # Step 5: Package results
    U = osqp_result.x
    y_model = R@U
    MSE = np.mean((y-y_model)**2)
    fit = DRT_Fit_Result(U, tau, offset=offset, cost=MSE, m=m)
    fit.y = y_model + offset
    
    return fit

def osqp_checkerboard_fit(time, y, tau, offset='Auto', checkerboard_iters=100):
    
    offset = y[-1] if offset == 'Auto' else offset
    U_values = np.zeros(len(tau))
    
    y_cap = y

    cap_U_values = np.zeros(len(tau))
    ind_U_values = np.zeros(len(tau))

    cap_offset = offset
    ind_offset = 0

    fits = []
    
    MSE = []
    cost = []
    
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
        fit = DRT_Fit_Result(U_values, tau, offset, 0, len(tau))
        
        # Calculate the cost 
        fit.predict_y(time)
        MSE_value = np.mean((y-fit.y)**2)
        MSE.append(MSE_value)
        fit.cost = MSE_value
        fits.append(fit)
        
    return fits