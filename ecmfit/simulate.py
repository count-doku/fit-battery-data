"""Simulate a battery cell, to produce data with known parameters.

Fitting code is hard to trust without a case where the right answer is known in
advance. This module runs the same ECM forward as a discrete-time state-space
model, so a pulse or a spectrum can be generated from a chosen parameter set and
the fit checked against it. That round trip is what ``tests/test_fit.py`` does.

Sign convention: current is positive on charge, negative on discharge.
"""

import numpy as np

from ecmfit.models import split_parameter

__all__ = ["ECMCell"]


class ECMCell:
    """An n-th order Thevenin cell, stepped at a fixed sample time.

    The state is ``[soc, u1, ..., un]``: state of charge plus the voltage across
    each RC element. Discretised exactly (zero-order hold on the current), so
    the sample time does not have to be small compared to the time constants::

        soc[k+1] = soc[k] + ts * eta * I / (capacity * 3600)
        u_i[k+1] = exp(-ts/tau_i) * u_i[k] + R_i * (1 - exp(-ts/tau_i)) * I
        V[k]     = OCV(soc[k]) + R0 * I + sum(u_i[k])

    Parameters
    ----------
    parameter : array-like
        ``[R0, R1..Rn, tau1..taun]`` — the same layout the fitting functions
        return, so a fitted result can be simulated directly.
    ocv : ecmfit.models.OCV
        Open-circuit voltage curve of the cell.
    capacity : float
        Nominal capacity in Ah.
    soc_init : float, optional
        Initial state of charge. Default 0.5.
    eta : float, optional
        Coulombic efficiency, applied on charge only. Default 1.0.
    ts : float, optional
        Sample time in s. Default 1.0.

    Examples
    --------
    >>> from ecmfit.models import OCV
    >>> ocv = OCV([0.0, 1.0], [2.5, 4.2])
    >>> cell = ECMCell([0.03, 0.04, 0.08, 10.0, 180.0], ocv, capacity=1.0)
    >>> t, v, soc = cell.simulate([-1.0] * 60 + [0.0] * 600)
    >>> bool(v[0] > v[1])  # voltage drops when the discharge starts
    True
    """

    def __init__(self, parameter, ocv, capacity, soc_init=0.5, eta=1.0, ts=1.0):
        self.r0, self.r, self.tau = split_parameter(parameter)
        self.ocv = ocv
        self.capacity = float(capacity)
        self.eta = float(eta)
        self.ts = float(ts)
        self.soc = float(soc_init)
        self.u = np.zeros_like(self.r)
        self._decay = np.exp(-self.ts / self.tau)

    def voltage(self, current):
        """Terminal voltage in V for ``current`` applied during the current step."""
        return float(self.ocv.voltage_at(self.soc) + self.r0 * current + self.u.sum())

    def step(self, current):
        """Advance the state by one sample time under ``current``."""
        eta = self.eta if current >= 0 else 1.0
        self.soc += self.ts * eta * current / (self.capacity * 3600.0)
        self.u = self._decay * self.u + self.r * (1.0 - self._decay) * current

    def simulate(self, current):
        """Run a current profile and return ``(t, voltage, soc)``.

        Parameters
        ----------
        current : array-like
            Current in A at each sample.

        Returns
        -------
        t : ndarray
            Time in s, starting at 0.
        voltage : ndarray
            Terminal voltage in V.
        soc : ndarray
            State of charge at each sample, before that sample's current is
            integrated.
        """
        current = np.asarray(current, dtype=float)
        voltage = np.empty(current.size)
        soc = np.empty(current.size)
        for k, i in enumerate(current):
            # Output before state update: V[k] depends on the state at k.
            soc[k] = self.soc
            voltage[k] = self.voltage(i)
            self.step(i)
        t = np.arange(current.size) * self.ts
        return t, voltage, soc
