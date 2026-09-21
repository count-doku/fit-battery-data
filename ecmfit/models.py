"""Equivalent-circuit model (ECM) equations in the time and frequency domain.

The model is an n-th order Thevenin network: a series resistance ``R0`` followed
by ``n`` parallel RC elements::

        ┌──R1──┐   ┌──R2──┐
    ──R0┤      ├───┤      ├── ...
        └──C1──┘   └──C2──┘

Each RC element is parameterised by its resistance ``Ri`` and its time constant
``tau_i = Ri * Ci`` rather than by ``Ci``, because ``tau`` is what a measurement
actually resolves and it keeps the parameters on comparable numeric scales for
the optimiser.

Parameter vectors are always laid out as::

    [R0, R1, ..., Rn, tau1, ..., taun]

so the model order follows from the length of the vector and the same functions
serve a 1st, 2nd or 3rd order model without change.

Sign convention: current is positive on charge, negative on discharge. All
voltages here are *overpotentials* — the open-circuit voltage is not included,
see :class:`OCV` for removing it from a measurement.
"""

import numpy as np

__all__ = [
    "OCV",
    "impedance",
    "split_parameter",
    "voltage_relaxation",
    "voltage_step_response",
]


def split_parameter(parameter):
    """Split ``[R0, R1..Rn, tau1..taun]`` into ``R0``, ``R`` and ``tau``.

    Parameters
    ----------
    parameter : array-like
        Flat parameter vector of odd length ``2n + 1``.

    Returns
    -------
    r0 : float
        Series resistance in Ohm.
    r : ndarray, shape (n,)
        RC resistances in Ohm.
    tau : ndarray, shape (n,)
        RC time constants in s.
    """
    p = np.asarray(parameter, dtype=float)
    if p.ndim != 1 or p.size < 3 or p.size % 2 == 0:
        raise ValueError(
            f"parameter must be a flat vector of odd length >= 3, got shape {p.shape}"
        )
    n = (p.size - 1) // 2
    return p[0], p[1 : 1 + n], p[1 + n :]


def voltage_step_response(parameter, current, t):
    """Overpotential of the ECM for a current applied from ``t = 0``.

    This is the response used to fit a *pulse*: the RC elements start relaxed
    and charge up with their respective time constants.

    Parameters
    ----------
    parameter : array-like
        ``[R0, R1..Rn, tau1..taun]``.
    current : array-like
        Applied current in A, same length as ``t``.
    t : array-like
        Time in s, starting at 0.

    Returns
    -------
    ndarray
        Overpotential in V.
    """
    r0, r, tau = split_parameter(parameter)
    current = np.asarray(current, dtype=float)
    t = np.asarray(t, dtype=float)
    rc = (r[:, None] * (1.0 - np.exp(-t / tau[:, None]))).sum(axis=0)
    return (r0 + rc) * current


def voltage_relaxation(parameter, current, t, t_pulse):
    """Overpotential of the ECM while relaxing after a preceding pulse.

    The RC elements start at the voltage they reached during a pulse of duration
    ``t_pulse`` at amplitude ``current[0]`` and decay from there. ``current`` is
    expected to hold that pulse amplitude in its first sample and zeros
    afterwards, so that the ohmic step ``R0 * I`` is still visible in the data —
    that step is what makes ``R0`` identifiable at all.

    Parameters
    ----------
    parameter : array-like
        ``[R0, R1..Rn, tau1..taun]``.
    current : array-like
        Current in A; first sample is the pulse amplitude, rest zero.
    t : array-like
        Time in s, starting at 0 at the end of the pulse.
    t_pulse : float
        Duration of the preceding pulse in s.

    Returns
    -------
    ndarray
        Overpotential in V.
    """
    r0, r, tau = split_parameter(parameter)
    current = np.asarray(current, dtype=float)
    t = np.asarray(t, dtype=float)
    # Voltage each RC element had reached at the end of the pulse.
    u_pulse = r * current[0] * (1.0 - np.exp(-t_pulse / tau))
    decay = (u_pulse[:, None] * np.exp(-t / tau[:, None])).sum(axis=0)
    return r0 * current + decay


def impedance(parameter, f):
    """Complex impedance of the ECM.

    Each RC element contributes ``Ri / (1 + j*omega*tau_i)``, which is the
    semicircle seen in a Nyquist plot.

    Parameters
    ----------
    parameter : array-like
        ``[R0, R1..Rn, tau1..taun]``.
    f : array-like
        Frequency in Hz.

    Returns
    -------
    ndarray of complex
        Impedance in Ohm.
    """
    r0, r, tau = split_parameter(parameter)
    jw = 2j * np.pi * np.asarray(f, dtype=float)
    return r0 + (r[:, None] / (1.0 + jw * tau[:, None])).sum(axis=0)


class OCV:
    """Open-circuit voltage curve, usable in both directions.

    Fitting needs the OCV twice and in opposite directions: once to turn a
    resting voltage into a state of charge, and once to subtract the OCV from a
    measured voltage so that only the overpotential is left. Keeping both on one
    object makes it impossible to accidentally call it the wrong way round.

    Parameters
    ----------
    soc : array-like
        State of charge grid, strictly increasing, dimensionless (0..1).
    voltage : array-like
        Open-circuit voltage in V at each grid point, strictly increasing.

    Examples
    --------
    >>> ocv = OCV([0.0, 1.0], [2.5, 4.2])
    >>> float(ocv.voltage_at(0.5))
    3.35
    >>> float(ocv.soc_at(3.35))
    0.5
    """

    def __init__(self, soc, voltage):
        self._soc = np.asarray(soc, dtype=float)
        self._voltage = np.asarray(voltage, dtype=float)
        if self._soc.shape != self._voltage.shape:
            raise ValueError("soc and voltage must have the same shape")
        if np.any(np.diff(self._soc) <= 0) or np.any(np.diff(self._voltage) <= 0):
            raise ValueError(
                "soc and voltage must both be strictly increasing so the curve "
                "can be inverted"
            )

    def voltage_at(self, soc):
        """Open-circuit voltage in V at a given state of charge."""
        return np.interp(soc, self._soc, self._voltage)

    def soc_at(self, voltage):
        """State of charge at a given resting (open-circuit) voltage in V."""
        return np.interp(voltage, self._voltage, self._soc)
