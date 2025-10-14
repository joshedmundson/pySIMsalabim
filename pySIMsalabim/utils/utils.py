import os,math
import pandas as pd
import scipy.integrate

def read_tj_file(session_path, tj_file_name='tj.dat'):
    """ Read relevant parameters for impedance of the tj file

    Parameters
    ----------
    session_path : string
        Path of the simulation folder for this session
    data_tj : dataFrame
        Pandas dataFrame containing the tj output file from ZimT

    Returns
    -------
    DataFrame
        Pandas dataFrame containing the time, voltage, current density and numerical error in the current density of the tj_file
    """

    data = pd.read_csv(os.path.join(session_path,tj_file_name), sep=r'\s+')

    return data

def get_integral_bounds(data, f_min=1e-2, f_max=1e6, f_steps=20, time_label='t'):
    """ Determine integral bounds in the time domain, used to compute the conductance and capacitance

    Parameters
    ----------
    data : dataFrame
        Pandas dataFrame containing the time, voltage, current density and numerical error in the current density of the tj_file
    time_label : string
        Specifies the heading used to denote the time column in data ('t' for tj.dat files, and 'time' for varFile.dat files)
    f_min : float
        Minimum frequency
    f_max : float
        Maximum frequency
    f_steps : float
        Frequency steps

    Returns
    -------
    list
        List of array indices that will be used in the plotting
    """

    # Total number of time points
    numTimePoints = len(data[time_label])

    # Check which time index corresponds to 1/fmax. We call this istart:
    istart = -1
    for i in range(numTimePoints):
        if math.isclose(data[time_label][i], 1/f_max, rel_tol = 2/f_steps): #note: don't use == to compare 2 floating points!
            istart = i

    # Starting time point could not be found.
    if istart == -1:
        msg = 'Could not find a time that corresponds to the highest frequency.'
        return -1, msg
    
    # print('Found istart: ', istart)

    # ifin: last index we should plot, corresponds to time = 1/f_min:
    ifin = numTimePoints - 1

    # isToPlot starts with istart:
    isToPlot = [istart]

    PlotRatio = max(1, round( (ifin-istart)/(math.log10(f_max/f_min) * f_steps)))

    # Incorrect plot ratio
    if PlotRatio < 1:
        msg = 'PlotRatio smaller than 1. It should at least be 1'
        return -1, msg

    # Then add the other indices:
    for i in range(istart+1, ifin-1):
        if (i-istart) % PlotRatio == 0: # note: % is python's modulo operator.
            isToPlot.append(i) # add the index to our array

    # Also include the last index:
    isToPlot.append(ifin)

    # Integral bounds have been determined, return the array with indices and a success message
    msg = 'Success'
    return isToPlot, msg

def integrate_fxt_dx(data, f_xt_name, x_name='x', t_name='time'):
    """ Integrates a column 'f_xt_name' within a pandas dataframe over 'x_name' for each unique value in 't_name'
        
    This function was designed with the format of varFile.dat in mind, which stores internal 
    variables, f(x,t), in the following way:

        x_1 f1(x_1, t_1) f2(x_1, t_1) f3(x_1, t_1) ... fN(x_1, t_1) t_1
        x_2 f1(x_2, t_1) f2(x_2, t_1) f3(x_2, t_1) ... fN(x_2, t_1) t_1 
        |
        x_M f1(x_M, t_1) f2(x_M, t_1) f3(x_M, t_1) ... fN(x_M, t_1) t_1
        x_1 f1(x_1, t_2) f2(x_1, t_2) f3(x_1, t_2) ... fN(x_1, t_2) t_2
        x_2 f1(x_2, t_2) f2(x_2, t_2) f3(x_2, t_2) ... fN(x_2, t_2) t_2
        |
        x_M f1(x_M, t_2) f2(x_M, t_2) f3(x_M, t_2) ... fN(x_M, t_2) t_2
        |
        x_1 f1(x_1, t_L) f2(x_1, t_L) f3(x_1, t_L) ... fN(x_1, t_L) t_L
        x_2 f1(x_2, t_L) f2(x_2, t_L) f3(x_2, t_L) ... fN(x_2, t_L) t_L
        |
        x_M f1(x_M, t_L) f2(x_M, t_L) f3(x_M, t_L) ... fN(x_M, t_L) t_L

    Where x_i is the ith position step in the whole material (including transport layers),
    t_j is the jth time step of the ZimT run, N is the maximum number of internal variables,
    M is the maximum number of position steps, and L is the maximum number of time steps

    Parameters
    ----------
    data : DataFrame
        DataFrame of dimensions ((M*L) x (N+2)) 
    f_xt_name : string
        The column name of the function to integrate over x
    x_name : string 
        The column name corresponding to the variable x
    t_name : string 
        The column name corresponding to the variable t

    Returns 
    -------
    DataFrame
        DataFrame containing two columns 't_name' and 'f_xt_name' of dimension (L x 2)
    """

    unique_time_vals = data[t_name].unique()

    f_t_data = pd.DataFrame(columns=[t_name, f_xt_name])

    for i in range(len(unique_time_vals)):
        # Find f(x,t=t_i) for the unique time step t_i in the simulated output 
        t_i = unique_time_vals[i]
        x = data[data[t_name]==t_i][x_name]
        f_xt_i = data[data[t_name]==t_i][f_xt_name]

        # Integrate f(x,t=t_i) over x 
        f_t_i = scipy.integrate.trapezoid(f_xt_i, x)

        # Store [t_i, f(t=t_i)] in f_t_data
        f_t_data.loc[i] = [t_i, f_t_i]
    
    return f_t_data

def update_cmd_pars(main_pars, cmd_pars):
    """Merges main parameters with command line parameters.

    Parameters
    ----------
    main_pars : list of dict
        A list containing dictionaries of the main parameters of the application. These parameters
        serve as the default values.
    cmd_pars : list of dict
        A list containing dictionaries of parameters passed through the command line interface.
        These parameters have higher precedence and override the main parameters in case
        of a conflict within any dictionary.

    Returns
    -------
    list of dict
        A list of dictionaries containing the combined set of parameters, with command line parameters
        overriding main parameters in case of conflicts within any dictionary.

    Raises
    ------
    ValueError
        If duplicate parameters are found in the command line parameters.
    """

    # check for duplicate in cmd_pars
    new_pars_names = []
    for line in cmd_pars:
        new_pars_names.append(line['par'])
    # if duplicates are found raise an error
    if len(new_pars_names) != len(set(new_pars_names)):
        raise ValueError('Duplicate parameters found in the command line parameters')

    # Update the main parameters with the command line parameters
    for par in cmd_pars:
        found = False
        for main in main_pars:
            if par['par'] == main['par']:
                main['val'] = str(par['val']) # convert to string
                break
        if not found:
            main_pars.append(par)          
    
    return main_pars

