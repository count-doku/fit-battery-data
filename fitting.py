import os

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize


def RC_equation_applied_current(parameter, I, t, t_pulse=False, plot=False):
    """
    Function to calculate voltage response to an applied current for a 
    3rd order ECM.

    Parameters
    ----------
    parameter : array-like
        Parameters passed to the model.
    I : array-like
        Current applied to the model.
    t : array-like
        Time array passed to the model.
    plot : BOOL, optional
        Plot output?. The default is False.

    Returns
    -------
    V_out : array-like
        Voltage response of the model to the applied current in given time.

    """
    R0 = parameter[0]
    R1 = parameter[1]
    R2 = parameter[2]
    R3 = parameter[3]
    tau1 = parameter[4]
    tau2 = parameter[5]
    tau3 = parameter[6]
    V_out = np.multiply(R0, I) + \
            np.multiply(R1*(1-np.exp(-t/tau1)), I) + \
            np.multiply(R2*(1-np.exp(-t/tau2)), I) + \
            np.multiply(R3*(1-np.exp(-t/tau3)), I)
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(t, V_out)
        ax.set(xlabel='t in s', ylabel='V in V', title='Voltage fit')
        
    return V_out


def RC_equation_relaxation(parameter, I, t, t_pulse, plot=False):
    """
    Equation to calculate the relaxation curve given a previous pulse for a 
    3rd order ECM.

    Parameters
    ----------
    parameter : array-like
        Parameters for the calculation.
    I : array-like
        Current array. Typical containing one current sample followed by zeros.
    t : array-like
        Time array. Typically starting with 0.
    t_pulse : scalar
        Time of the previous pulse.
    plot : bool, optional
        Plot the result?. The default is False.

    Returns
    -------
    V_out : array-like
        Output voltage array.
        
    """
    R0 = parameter[0]
    R1 = parameter[1]
    R2 = parameter[2]
    R3 = parameter[3]
    tau1 = parameter[4]
    tau2 = parameter[5]
    tau3 = parameter[6]
    
    V_est1 = R1 * I[0] * (1-np.exp(-t_pulse/tau1))
    V_est2 = R2 * I[0] * (1-np.exp(-t_pulse/tau2))
    V_est3 = R3 * I[0] * (1-np.exp(-t_pulse/tau3))
    
    V_out = np.multiply(R0, I) + \
            V_est1*np.exp(-t/tau1) + \
            V_est2*np.exp(-t/tau2) + \
            V_est3*np.exp(-t/tau3)
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(t, V_out)
        ax.set(xlabel='t in s', ylabel='V in V', title='Voltage fit')
        
    return V_out
    
    
def objective_function(parameter, V_truth, model, I, t, t_pulse=None):
    """
    Calculates rmse as objective for the minimization.
    
    Parameters
    ----------
    parameter : array-like
        Parameters passed to the model.
    V_truth : array-like
        Voltage signal representing the truth.
    model : function
        Function calculating the model output voltage.
    I : array-like
        Input current passed to the model.
    t : array-like
        Time array passed to the model.
    t_pulse : scalar, optional
        Time of the previous charge/discharge pulse. Only for relaxation fitting.
        The default is None.
    
    Returns
    -------
    e : scalar
        Root-mean-squared-error minimized by the optimizer.
    
    """
    # Calculate fitting function
    V_out = model(parameter, I, t, t_pulse=t_pulse)
    
    e = np.sqrt(np.mean(np.subtract(V_out, V_truth)**2))

    return e


def fit_on_pulse(pulse:dict, ocv_map:object, CN:float, model:object,
                 objective_function:object,
                 bounds=None, plot=False, hold_R0=False, plot_path=None):
    """
    Fit <model> to <pulse> and return specified model parameter.
    
    The pulse need to be a dict containing the keys: 'I', 'V' and 't' for 
    current, voltage and time of the pulse, respectively. For a proper estimation
    of the internal resistance, the pulse should contain the current and voltage 
    step at the beginning of the pulse. For the initial state-of-charge
    determination, the first voltage sample should be after relaxation and of
    zero-current.
    The function acts as following:
        1) Extract data arrays from pulse dictionary
        2) Check whether it is a discharge- or charge pulse
        3) Get pulse initial SOC from <ocv_map>
        4) Normalize pulse time s.t. it starts with 0
        5) Calculate time diff for the calculation of the charge
        6) Calculate charge of the pulse for SOC calculation
        7) Get OCV from <ocv_map> using SOC
        8) Subtract OCV from pulse voltage and remove constant offset
        9) Guess initial parameters (some more sophisticated method could be 
                                     inserted here)
        10) If <hold_R0>: Force R0 estimation for the fitting process
        11) Fit pulse with Nelder-Meads and provided bounds
        12) Write parameter array [SOC, P0, P1, P2, P3 ... PN]
        13) If <plot>: Plot fitting result and if <plot_path> save figure on
                                     specified path
                                     
    Parameters
    ----------
    pulse : dict
        DESCRIPTION.
    ocv_map : object
        DESCRIPTION.
    CN : float
        DESCRIPTION.
    model : object
        DESCRIPTION.
    objective_function : object
        DESCRIPTION.
    ocv_interp : object
        DESCRIPTION.
    bounds : TYPE, optional
        DESCRIPTION. The default is None.
    plot : TYPE, optional
        DESCRIPTION. The default is False.
    hold_R0 : TYPE, optional
        DESCRIPTION. The default is False.
    plot_path : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    pulse_parameter : TYPE
        DESCRIPTION.

    """
    # Extract data
    pulse_current = pulse['I']
    pulse_voltage = pulse['V']
    pulse_time = pulse['t']
    
    # Check if charging or discharging pulse
    if np.mean(pulse_current) < 0:
        pulse_type = 'discharge'
    else:
        pulse_type = 'charge'
    print(f'\t Start fitting on {pulse_type} pulse')   # Verbose
    
    # Obtain initial SOC from ocv_map and initial pulse voltage
    pulse_SOC_init = ocv_map(pulse_voltage[0]).item()
    
    # Normalize pulse time and calculate sample time (diff)
    pulse_time = pulse_time - pulse_time[0]
    pulse_time_diff = np.diff(pulse_time)
    
    # Calculate charge of pulse
    # The first value is 0 and the second value is the sample time times the
    # applied current.
    pulse_charge = [0]
    for I, t, i in zip(pulse_current[:-1], pulse_time_diff,
                       range(len(pulse_current[:-1]))):
        pulse_charge.append(pulse_charge[i] + I*t)
    
    # Calculate SOC of pulse and subtract OCV from pulse voltage
    pulse_charge = np.array(pulse_charge)
    pulse_soc = pulse_SOC_init + 1/CN/3600 * pulse_charge
    pulse_ocv = ocv_map(pulse_soc)
    pulse_voltage_detrend = pulse_voltage - pulse_ocv
    pulse_voltage_detrend = pulse_voltage_detrend - pulse_voltage_detrend[0]
    
    # Guess initial parameters
    p_init = [pulse_voltage_detrend[1]/pulse_current[1],
              pulse_voltage_detrend[-1]/pulse_current[-1],
              pulse_voltage_detrend[-1]/pulse_current[-1],
              pulse_voltage_detrend[-1]/pulse_current[-1],
              10, 180, 600]
    
    # Determine R0 from first samples and hold it
    if hold_R0:
        bounds[0] = (p_init[0], p_init[0])
        
    # Fit pulse
    p_res = minimize(objective_function,
                      p_init,
                      args=(pulse_voltage_detrend,
                            model,
                            pulse_current,
                            pulse_time),
                      bounds=bounds,
                      method='Nelder-Mead')  
    p = p_res['x']
    
    # Allocate paramter array [SOC, p1, p2, ... pn]
    pulse_parameter = np.zeros(len(p_init)+1)
    
    # Write parameters in array
    pulse_parameter[0] = pulse_SOC_init
    pulse_parameter[1:] = p
    
    if plot:
        model(p, pulse_current, pulse_time, plot=True)
        plt.plot(pulse_time, pulse_voltage_detrend, linestyle='--')
        ax = plt.gca()
        ax.set(title=f'SOC: {pulse_SOC_init}')
        ax.legend(['fit', 'truth'])
        print(f'SOC: {pulse_SOC_init}\n Parameter: {p}')
        plt.tight_layout()
        if plot_path:
            file_name = str(pulse_SOC_init)+'.pdf'
            path = os.path.join(plot_path, file_name)
            plt.savefig(path)
            plt.close()
            
    return pulse_parameter


def fit_on_relaxation(pulse:dict, relax:dict, ocv_map:object, CN:float,
                      model:object, objective_function:object,
                      bounds=None, plot=False, hold_R0=False, plot_path=None):
    """
    Fit <model> to <relax> using preceding <pulse> and return specified model
    parameter.
    
    The pulse and the relax information need to be a dict conatining the keys:
    'I', 'V' and 't' for current, voltage and time of the steps, respectively.
    For a proper estimation of the internal resistance, the relaxation should 
    contain the current and voltage step at the beginning of the relaxation.
    For the initial state-of-charge determination, the first voltage sample of
    <pulse> should be after relaxation and of zero-current. 
    The function acts as following:
        1) Extract data arrays from pulse and relax dict
        2) Check whether it is a discharge- or charge relaxation
        3) Get pulse initial SOC from <ocv_map>
        4) Normalize relaxation time s.t. it starts with 0
        5) Calculate SOC at the end of the preceding pulse
        6) Calculate relaxation OCV
        7) Subtract OCV from given relaxation voltage
        8) Guess initial parameters (some more sophisticated method could be 
                                     inserted here)
        9) If <hold_R0>: Force R0 estimation for the fitting process. If R0
            estimation is below 1 mOhm, take next sample. Repeat this 8 times.
        10) Fit pulse with Nelder-Meads and provided bounds
        11) Write parameter array [SOC, P0, P1, P2, P3 ... PN]
        12) If <plot>: Plot fitting result and if <plot_path> save figure on
                       specified path
    
    Parameters
    ----------
    pulse : dict
        DESCRIPTION.
    relax : dict
        DESCRIPTION.
    ocv_map : object
        DESCRIPTION.
    CN : float
        DESCRIPTION.
    model : object
        DESCRIPTION.
    objective_function : object
        DESCRIPTION.
    bounds : TYPE, optional
        DESCRIPTION. The default is None.
    plot : TYPE, optional
        DESCRIPTION. The default is False.
    hold_R0 : TYPE, optional
        DESCRIPTION. The default is False.
    plot_path : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    relax_parameter : TYPE
        DESCRIPTION.

    """
    # Extract data
    pulse_current = pulse['I']
    relax_current = relax['I']
    pulse_voltage = pulse['V']
    relax_voltage = relax['V']
    pulse_time = pulse['t']
    relax_time = relax['t']
    
    # Check if charging or discharging relaxation
    if np.mean(pulse_current) < 0:
        pulse_type = 'discharge'
    else:
        pulse_type = 'charge'
    print(f'\t Start fitting on {pulse_type} relaxation')   # Verbose
    
    # Obtain initial SOC from ocv_map and initial pulse voltage
    pulse_SOC_init = ocv_map(pulse_voltage[0]).item()
    
    # Normalize relaxation time
    relax_time = relax_time - relax_time[0]
    
    # Calculate SOC at the end of the pulse
    pulse_duration = pulse_time[-1] - pulse_time[0]
    pulse_current_amp = relax_current[0]
    relax_SOC_init = pulse_SOC_init + \
        1/CN/3600 * pulse_current_amp * pulse_duration
    
    # Calculate relaxation SOC and subtract OCV from relaxation voltage
    relax_ocv = ocv_map(relax_SOC_init)
    relax_voltage_detrend = relax_voltage - relax_ocv
    
    # Estimate time constants
    tau1_est = 10
    tau2_est = 180
    tau3_est = 600
    
    # Estimate resistances
    R0_est = (relax_voltage_detrend[0] - relax_voltage_detrend[1])/pulse_current_amp
    R1_est = relax_voltage_detrend[0]\
        /relax_current[0]*(1-np.exp(-pulse_duration/tau1_est))
    R2_est = relax_voltage_detrend[0]\
        /relax_current[0]*(1-np.exp(-pulse_duration/tau2_est))
    R3_est = relax_voltage_detrend[0]\
        /relax_current[0]*(1-np.exp(-pulse_duration/tau3_est))
        
    p_init = [R0_est, R1_est, R2_est, R3_est, tau1_est, tau2_est, tau3_est]
    
    # Determine R0 from first samples and hold ist
    if hold_R0:
        for i in range(2, 10):
            if R0_est <= 1e-3:
                R0_est = (relax_voltage_detrend[0] - relax_voltage_detrend[i])/pulse_current_amp
                print(f'Bad R0: {R0_est}; Took next sample for estimation')
            else:
                continue
        bounds[0] = (R0_est, R0_est)
        
    p_res = minimize(objective_function,
                      p_init,
                      args=(relax_voltage_detrend,
                            model,
                            relax_current,
                            relax_time,
                            pulse_duration),
                      bounds=bounds,
                      method='Nelder-Mead')
    p = p_res['x']
    
    # Allocate parameter array [SOC, p1, p2, ... pn]
    relax_parameter = np.zeros(len(p_init)+1)
    
    # Write parameters in array
    relax_parameter[0] = pulse_SOC_init
    relax_parameter[1:] = p
    
    if plot:
        model(p, relax_current, relax_time, pulse_duration, plot=True)
        plt.plot(relax_time, relax_voltage_detrend, linestyle='--')
        ax = plt.gca()
        ax.set(title=f'SOC: {relax_SOC_init}')
        ax.legend(['fit', 'truth'])
        print(f'SOC: {relax_SOC_init}\n Parameter: {p}')
        plt.tight_layout()
        if plot_path:
            file_name = str(pulse_SOC_init)+'.pdf'
            path = os.path.join(plot_path, file_name)
            plt.savefig(path)
            plt.close()
        
    return relax_parameter