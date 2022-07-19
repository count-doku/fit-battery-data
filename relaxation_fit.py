import matplotlib.pyplot as plt
import numpy as np

from utils.battery_models import batteryModel_static
from scipy import interpolate


def RC_equation_relaxation(parameter, I, t, t_pulse, plot=False):
    """
    Equation to calculate the relaxation curve given a previous pulse for a 
    2nd order ECM.

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
    tau1 = parameter[3]
    tau2 = parameter[4]
    
    V_est1 = R1 * I[0] * (1-np.exp(-t_pulse/tau1))
    V_est2 = R2 * I[0] * (1-np.exp(-t_pulse/tau2))
    
    V_out = np.multiply(R0, I) + \
            V_est1*np.exp(-t/tau1) + \
            V_est2*np.exp(-t/tau2)
    
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


if __name__ == '__main__':
    
    parameter = [0.03, 0.04, 0.08, 10, 180]
    
    bm = batteryModel_static(R0=parameter[0],
                             R1=parameter[1],
                             R2=parameter[2],
                             tau1=parameter[3],
                             tau2=parameter[4],
                             ocv=[2.5, 4.2],
                             grid_soc=[0, 1],
                             eta=1,
                             ts=0.1,
                             soc_init=0.4,
                             u1_init=0,
                             u2_init=0)
    
    ocv_interp = interpolate.interp1d([0, 1], [2.5, 4.2])
    
    I = [-1]*600 + [0]*19999   
    t = np.arange(0, len(I)*0.1, 0.1)
    y_log = []
    x_log = []
    x = bm.state
    for u in I: 
        x = bm.state_equation(u, x)
        y = bm.output_equation(u, x)
        bm.update_parameter(x[0])
        y_log.append(y.item())
        x_log.append(x)
    
    V_relax = y_log[599:]
    I_relax = I[599:]
    t_relax = t[599:]
    t_relax = t_relax - t_relax[0]
    
    t_pulse = 60
    I_pulse = I_relax[0]
    soc_init_pulse = 0.4
    
    soc_init_relax = soc_init_pulse + 1/3600 * t_pulse * I_pulse
    ocv_relax = ocv_interp(soc_init_relax)
    V_relax_det = V_relax - ocv_relax
    
    tau1_est = 10
    tau2_est = 180
    
    R0_est = (V_relax_det[0] - V_relax_det[1])/I_pulse
    R1_est = V_relax_det[0]/(I_pulse*(1-np.exp(-t_pulse/tau1_est)))
    R2_est = V_relax_det[0]/(I_pulse*(1-np.exp(-t_pulse/tau2_est)))
    
    p_init = [R0_est, R1_est, R2_est, tau1_est, tau2_est]
    
    
    V_est_init = RC_equation_relaxation(p_init, I_relax, t_relax, t_pulse)
    V_est_perfect = RC_equation_relaxation(parameter, I_relax, t_relax, t_pulse)
    
    #TODO: Add fitting here
    
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0,0].plot(t_relax, I_relax, label='I')
    ax[0,1].plot(t_relax, V_est_init, label='V est init')
    ax[0,1].plot(t_relax, V_relax_det, label='V true')
    ax[0,1].plot(t_relax, V_est_perfect, '--', label='V est perfect')
    
    [ax[i,j].legend() for i in range(3) for j in range(3)]