"""Find System Lifetimes using Distribution of Relaxation Times (DRT) Fitting in the Time Domain"""
######### Package Imports #########################################################################

import matplotlib.pyplot as plt
# from pySIMsalabim.utils import device_parameters as utils_dev
import scipy.optimize as so
import numpy as np
import torch

######### References ##############################################################################

# [1] M. Schönleber, D. Klotz, and E. Ivers-Tiffée, ‘A Method for Improving the Robustness of linear 
# Kramers-Kronig Validity Tests’, Electrochimica Acta, vol. 131, pp. 20–27, June 2014, 
# doi: 10.1016/j.electacta.2014.01.034.

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

    def set_DRT_curve(self, t, backend='numpy', device=torch.device('cpu')):
        """ Sets self.DRT_curve = DRT_curve(t) using the objects attributes

        Parameters
        ----------
        t : float or list/numpy.ndarray, shape (n,)
            Time values to calculate the DRT_curve curve
        backend : {'numpy', 'torch'} (optional)
            Determines whether matrix operations are done with numpy or pytorch. Default numpy. 
        device : torch.device (optional)
            Determines what device is used for matrix ops if backend='torch'.
        
        Returns
        -------
        None
        """
        self.DRT_curve = DRT_curve(t, self.U, self.tau, self.offset, backend=backend, device=device)


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
            self.upper_bound = bounds[0]
        else:
            self.lower_bound = None
            self.upper_bound = None
        
    def forward(self, t):
        if self.lower_bound is not None:
            params_clamped = torch.clamp(self.params, self.lower_bound, self.upper_bound)
        else:
            params_clamped = self.params
        return DRT_curve_pytorch(t, params_clamped[:-1], self.tau, params_clamped[-1])


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
#########################################################

def DRT_curve_numpy(t, U, tau, offset=0):
    """Use numpy.ndarray objects to calculate the function 
        DRT_curve(t) = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2) + ... + U_m*exp(-t/tau_m) + offset
    
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
        Initial guess of the DRT_curve offset value
    device : torch.device 
        The specified device for torch based computation if backend='torch'. 
    
    Returns 
    -------
    calculated_DRT_curve : numpy.ndarray with shape (n,)
        DRT_curve at all time values in t
    """
    # Calculate DRT_curve(t) for all values in t
    calculated_DRT_curve = (U @ np.exp(-np.outer(1/tau, t))) + offset

    return calculated_DRT_curve


def DRT_curve_pytorch(t, U, tau, offset=0):
    """Use torch.Tensor objects to calculate the function
        DRT_curve(t) = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2) + ... + U_m*exp(-t/tau_m) + offset
    
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
        Initial guess of the DRT_curve offset value
    
    Returns 
    -------
    calculated_DRT_curve : torch.Tensor with shape (n,)
        DRT_curve at all time values in t
    """

    # Calculate DRT_curve(t) for all values in t
    calculated_DRT_curve = (U @ torch.exp(-torch.outer(1/tau, t))) + offset
    
    return calculated_DRT_curve


def DRT_curve(t, U, tau, offset=0, backend='numpy', device=torch.device('cpu')):
    """Calculate the function
        DRT_curve(t) = U_1*exp(-t/tau_1) + U_2*exp(-t/tau_2) + ... + U_m*exp(-t/tau_m) + offset
    
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
        Initial guess of the DRT_curve offset value
    backend : {'numpy', 'torch'} (optional)
        Determines whether matrix operations are done with numpy or pytorch. Default numpy. 
    device : torch.device (optional)
        Determines what device is used for matrix ops if backend='torch'.
    
    Returns 
    -------
    calculated_DRT_curve : numpy.ndarray or torch.Tensor with shape (n,)
        DRT_curve at all time values in t
    """
    
    if backend == 'numpy':
        t = numpy_converter(t)
        U = numpy_converter(U)
        tau = numpy_converter(tau)

        return DRT_curve_numpy(t, U, tau, offset=offset)

    elif backend == 'torch':
        t = torch_tensor_converter(t, device=device)
        U = torch_tensor_converter(U, device=device)
        tau = torch_tensor_converter(tau, device=device)

        return DRT_curve_pytorch(t, U, tau, offset)


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
        U0 = np.ones(m)*U_scale_factor/m
        tau0 = np.geomspace(t[0], t[-1], m)
        offset0 = offset0
        
        # Fit the curve
        fit = fit_DRT_curve(t, y, U0, tau0, offset0=offset0, set_DRT_curve=set_DRT_curve, **kwargs)
        
        # Add to fits
        fits.append(fit)
        
    return fits


def fit_DRT_curve_linear(t, y, tau, U_scale_factor=1, offset=0, alpha=0, set_DRT_curve=True, backend='numpy', device='cpu', **kwargs):
    """Fits an DRT_curve(t) curve to a function y(t) using a grid-based approach

        Uses a linear approach to fit 
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
        tau : list/numpy.ndarray with shape (m,)
            Grid of tau values used in calculating DRT_curve(t)
        U_scale_factor : float
            Determines the magnitude and polarity of initial guess for U
        offset0 : float (optional)
            Initial guess of the DRT_curve offset value
        set_DRT_curve : bool (optional)
            If true, will set the DRT_curve attribute of the returned DRT_Fit_Result object using t
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
                    Number of lifetimes/coefficients used in DRT_curve, given by the length of U/tau
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
        error = lambda x : y - DRT_curve(t, x[:-1], tau, offset=x[-1], backend='numpy')
        
        # Minimise the error function using scipy
        fit = so.least_squares(error, x0=x0, **kwargs)

        # Store the results in a DRT_Fit object
        drt_fit = DRT_Fit_Result(fit.x[:-1], tau, fit.x[-1], fit.cost, m)

        # Set the DRT_curve curve for DRT_Fit_Result result if calc_DRT_curve=True
        drt_fit.set_DRT_curve(t) if set_DRT_curve else None

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

        error = lambda x : y - DRT_curve(t, x[:-1], tau, offset=x[-1], backend='torch', device=device)

        # Minimise the error function using scipy
        fit = so.least_squares(error, x0=x0, **kwargs)

        # Store the results in a DRT_Fit object
        drt_fit = DRT_Fit_Result(fit.x[:-1], tau, fit.x[-1], fit.cost, m)

        # Set the DRT_curve curve for DRT_Fit_Result result if calc_DRT_curve=True
        drt_fit.set_DRT_curve(t) if set_DRT_curve else None

        # Assume params is of dimension 2 x m, where m is the number of tau/U_values
        return drt_fit


def fit_DRT_curve_checkerboard(t, y, tau, U_scale_factor=1, offset=0, set_DRT_curve=True, backend='numpy', device='cpu', **kwargs):
    max_iters = 10 
    # NOTE: we probably want to set the offset and scale factor guesses ourselves
    for i in range(len(max_iters)):
        cap_fit = fit_DRT_curve_linear(t, y, tau, U_scale_factor=U_scale_factor, offset=offset, 
                                       set_DRT_curve=set_DRT_curve, backend=backend, device=device, bounds=(0, np.inf), **kwargs)
        
        # Modify the values you then want to fit to 
        y = y - cap_fit.DRT_curve

        # Fit inductive curve
        cap_fit = fit_DRT_curve_linear(t, y, tau, U_scale_factor=U_scale_factor, offset=offset, 
                                       set_DRT_curve=set_DRT_curve, backend=backend, device=device, bounds=(-np.inf, 0), **kwargs)
        
        y = y - cap_fit.DRT_curve
        
        
def fit_DRT_curve_linear_torch(t, y, tau, U_scale_factor='Auto', offset='Auto', set_DRT_curve=True, alpha=0, max_step_iter=20, max_iter=200, device='cpu', bounds=None, **kwargs):
    
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
    
    def closure():
        optimizer.zero_grad()
        predictions = linear_model(t)
                
        # Define loss using our IC regulariser 
        loss = torch.sum((predictions - y)**2) + alpha*IC_ratio_reg(linear_model.params[:-1], alpha=alpha, backend='torch', device=device)
        
        loss.backward()
        loss_history.append(loss.item())
        return loss
    
    for epoch in range(max_iter // max_step_iter):
        optimizer.step(closure)
        
        if len(loss_history) > 1:
            if abs(loss_history[-1] - loss_history[-2]) < 1e-9:
                break
    
    # Return results 
    final_loss = loss_history[-1]
    final_params = linear_model.params.detach().cpu().numpy()
    U = final_params[:-1]
    offset = final_params[-1]
    
    final_fit_result = DRT_Fit_Result(U, tau, offset, final_loss, m=len(tau))
    
    if set_DRT_curve:
        final_fit_result.set_DRT_curve(t)
    
    return final_fit_result