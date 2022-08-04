import numpy as np
import matplotlib.pyplot as plt

from fitting import impedance_model, fit_on_eis, objective_function_impedance

r0 = 0.009
r1 = 0.01
r2 = 0.03
tau1 = 10
tau2 = 100
r3 = 0.02
L3 = 1e-6

c1 = tau1/r1
c2 = tau2/r2

f = np.logspace(-3, 3, 100)
omega = 2*np.pi*f

Z = r0 + 1/(1/r1+1j*omega*c1) + 1/(1/r2+1j*omega*c2) + 1/(1/r3+1/(1j*omega*L3))

fig, ax = plt.subplots()
ax.plot(np.real(Z), -np.imag(Z))
ax.set(xlabel='real(Z)', ylabel='imag(Z)', title='spectra')
ax.axis('equal')

Z = {'frequency': f,
      'real': np.real(Z),
      'imag': np.imag(Z)}



p = fit_on_eis(Z, impedance_model, objective_function_impedance, soc=10, plot=True)