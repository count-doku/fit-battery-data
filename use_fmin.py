import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin


def model_(I, ts,  plot=False):
    """
    Model to create an artificial ralaxation curve for this example.
    
    I: Input current array
    ts: Sample time
    plot: Plot output?
    """
    
    R0 = 0.015
    R1 = 0.025
    R2 = 0.035
    R3 = 0.04
    C1 = 100
    C2 = 1000
    C3 = 2000
    tau1 = R1*C1
    tau2 = R2*C2
    tau3 = R3*C3
    _len = len(I)
    t = np.arange(0, _len*ts, ts)
    #print(_len)
    #print(len(t))
    V_out = np.multiply(R0, I) + \
            np.multiply(R1*(1-np.exp(-t/tau1)), I) + \
            np.multiply(R2*(1-np.exp(-t/tau2)), I) + \
            np.multiply(R3*(1-np.exp(-t/tau3)), I)
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(t, V_out)
        ax.set(xlabel='t in s', ylabel='V in V', title='Relaxation voltage truth')
        
        
    return V_out, t


def model(parameter, I, ts,  plot=False):
    """
    Model to calculate the voltage curve of the fitted parameters.
    
    parameter: [R0, R1, R2, tau1, tau2]
    I: Input current array
    ts: Sample time
    """
    
    R0 = parameter[0]
    R1 = parameter[1]
    R2 = parameter[2]
    tau1 = parameter[3]
    tau2 = parameter[4]
    _len = len(I)
    t = np.arange(0, _len*ts, ts)
    V_out = np.multiply(R0, I) + \
            np.multiply(R1*(1-np.exp(-t/tau1)), I) + \
            np.multiply(R2*(1-np.exp(-t/tau2)), I)
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(t, V_out)
        ax.set(xlabel='t in s', ylabel='V in V', title='Relaxation voltage fit')
        
        
    return V_out, t


def objective_function(parameter, V_truth, model, I, ts):
    """
    Objective function for this example. Calculates the rmse of truth and fit.
    
    parameter: [R0, R1, R2, tau1, tau2]
    V_truth: Voltage curve to compare with
    I: Current signal
    ts: Sample time in s
    """
    
    # Calculate fitting function
    V_out, _ = model(parameter, I, ts)
    
    e = np.sqrt(np.mean(np.subtract(V_out, V_truth)**2))

    return e


if __name__ == '__main__':
    # Create current signal
    I = np.ones(500)  # 500 samples
    ts = 1  # Sample time: 1s
    
    # Set initial guess
    x_init = [0.015, 0.025, 0.035, 1.5, 25]  # This is a very good initial guess
    
    # Calculate the true voltage curve
    V_truth, t = model_(I, ts, plot=False)
    
    # Fit model to true voltage curve
    x = fmin(objective_function, x_init, args=(V_truth, model, I, ts), maxiter=None)
    
    # Calculate fittet model voltage curve
    V_fit, t = model(x, I, ts, plot=False)
    
    # Print result
    print(f'Fittet parameters: \n R0: {x[0]} \n R1: {x[1]} \n R2: {x[2]} \n tau1: {x[3]} \n tau2: {x[4]}')
    
    # Plot result
    fig, ax = plt.subplots()
    ax.plot(t, V_truth, label='$V_{truth}$')
    ax.plot(t, V_fit, label='$V_{fit}$', linewidth=1, linestyle='--')
    ax.set(xlabel='time in s', ylabel='voltage in V', title='Comparison truth - fit')
    ax.legend()
