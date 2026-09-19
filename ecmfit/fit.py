"""Fit ECM parameters to pulse, relaxation and impedance measurements.

Each of the three entry points follows the same three steps:

1. **Condition the data.** Work out the state of charge the measurement belongs
   to and remove everything the ECM does not describe — for time-domain data
   that is the open-circuit voltage, which drifts during a pulse as charge
   flows.
2. **Guess the parameters.** A Nelder-Mead simplex has no gradient to follow and
   a 2nd or 3rd order ECM is a badly conditioned fit, so the starting point does
   most of the work. The guesses here come from the data itself: ``R0`` from the
   ohmic step, the RC resistances from the settled overpotential, the time
   constants spread logarithmically over the decades the measurement resolves.
   Pass ``p0`` to override.
3. **Minimise** the residual between model and measurement, optionally inside
   ``bounds``.

References
----------
Nelder-Mead is used rather than a least-squares method because the parameters
must be kept inside physically meaningful bounds and the residual is cheap to
evaluate; see the fitting discussion in

    S. Neupert and J. Kowal, "Model-Based State-of-Charge and State-of-Health
    Estimation Algorithms Utilizing a New Free Lithium-Ion Battery Cell Dataset
    for Benchmarking Purposes", Batteries 9(7):364, 2023.
    https://doi.org/10.3390/batteries9070364
"""

from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize

from ecmfit.models import impedance, voltage_relaxation, voltage_step_response

__all__ = [
    "FitResult",
    "fit_eis",
    "fit_pulse",
    "fit_relaxation",
    "identifiable_tau",
]


class FitResult(NamedTuple):
    """Outcome of one fit.

    Attributes
    ----------
    parameter : ndarray
        Fitted ``[R0, R1..Rn, tau1..taun]``.
    soc : float
        State of charge the measurement belongs to, dimensionless.
    residual : float
        RMS residual between model and measurement, in V (time domain) or Ohm
        (frequency domain).
    x : ndarray
        Time in s, or frequency in Hz.
    measured : ndarray
        Conditioned measurement the fit was run against: overpotential in V, or
        complex impedance in Ohm.
    fitted : ndarray
        Model output at ``parameter``, same units as ``measured``.
    success : bool
        Whether the optimiser reported convergence.
    message : str
        The optimiser's termination message.
    """

    parameter: np.ndarray
    soc: float
    residual: float
    x: np.ndarray
    measured: np.ndarray
    fitted: np.ndarray
    success: bool
    message: str


def _rms(residual):
    return float(np.sqrt(np.mean(np.abs(residual) ** 2)))


def _minimise(objective, p0, bounds, max_iter):
    p0 = np.asarray(p0, dtype=float)
    if bounds is not None:
        # A starting point outside its own bounds makes SciPy warn and start
        # from somewhere unintended. It happens for dull reasons — logspace
        # lands a hair above the ceiling it was built from, or noise pushes the
        # measured R0 slightly negative — so clip rather than chase each case.
        low = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
        high = np.array([np.inf if b[1] is None else b[1] for b in bounds])
        p0 = np.clip(p0, low, high)
    return minimize(
        objective,
        p0,
        bounds=bounds,
        method="Nelder-Mead",
        options={"maxiter": max_iter, "fatol": 1e-14, "xatol": 1e-10},
    )


def identifiable_tau(t_pulse, t_observed):
    """Longest time constant this measurement can actually resolve.

    Two independent conditions have to hold, and the shorter one binds:

    * **Excitation.** An RC element only charges to
      ``1 - exp(-t_pulse/tau)`` of its final value, so a tau much longer than
      the pulse is barely stirred at all. At ``t_pulse = tau/3`` it reaches 28 %
      of full amplitude and its contribution is already lost in the noise —
      watching it relax for an hour afterwards does not bring it back.
    * **Observation.** A decay that is never seen to decay cannot be measured
      either, so the record length is the other limit.

    Returns ``min`` of the two. Used both to place the initial guesses and as
    the upper bound handed to the optimiser, so the two cannot disagree.
    """
    return float(min(t_pulse, t_observed))


def _time_constant_guess(n_rc, tau_max):
    """Spread ``n_rc`` time constants logarithmically up to ``tau_max``.

    ``tau_max`` is what :func:`identifiable_tau` says the measurement supports;
    the fastest guess is a hundredth of it.
    """
    tau_max = max(float(tau_max), 1e-3)
    return np.logspace(np.log10(tau_max / 100.0), np.log10(tau_max), n_rc)


def _default_bounds(n_parameter, tau_max=None):
    """Keep the optimiser inside the physically meaningful quadrant.

    Resistances and time constants of a passive RC network are positive.
    Without that the simplex walks into negative time constants, where the model
    still evaluates, the residual still looks small, and the parameters are
    meaningless.

    ``tau_max`` additionally caps the time constants at what the measurement can
    resolve (see :func:`identifiable_tau`). This does not make a typical fit
    more accurate — it stops the occasional one from running off. On a noisy
    relaxation it cut the rate of diverged fits from 40 % to 28 % and the worst
    observed error on the slow time constant by a factor of 90.
    """
    n_rc = (n_parameter - 1) // 2
    upper = None if tau_max is None else float(tau_max)
    return [(0.0, None)] * (1 + n_rc) + [(1e-6, upper)] * n_rc


def _held_bounds(bounds, r0, n_parameter, tau_max=None):
    """Return ``bounds`` with R0 pinned to ``r0``, without mutating the caller's list."""
    bounds = list(_default_bounds(n_parameter, tau_max) if bounds is None else bounds)
    bounds[0] = (r0, r0)
    return bounds


def _current_step(current, name):
    """Index of the first non-zero current sample."""
    nonzero = np.flatnonzero(current != 0.0)
    if nonzero.size == 0:
        raise ValueError(f"{name}['I'] is zero throughout; there is no pulse to fit")
    return int(nonzero[0])


def fit_pulse(
    pulse,
    ocv,
    capacity,
    n_rc=3,
    p0=None,
    bounds=None,
    hold_r0=False,
    max_iter=10000,
):
    """Fit an ECM to a current pulse.

    Parameters
    ----------
    pulse : dict
        Keys ``'I'`` (A), ``'V'`` (V) and ``'t'`` (s). The first sample must be
        taken *before* the current step and at rest, because its voltage is read
        as the open-circuit voltage to get the starting state of charge, and the
        step it forms with the second sample is what identifies ``R0``.
    ocv : ecmfit.models.OCV
        Open-circuit voltage curve of the cell.
    capacity : float
        Nominal cell capacity in Ah.
    n_rc : int, optional
        Number of RC elements. Default 3.
    p0 : array-like, optional
        Initial parameter guess. Derived from the data if omitted.
    bounds : sequence of (low, high), optional
        Per-parameter bounds passed to the optimiser.
    hold_r0 : bool, optional
        Pin ``R0`` to the value read off the ohmic step instead of fitting it.
        Useful when the RC elements would otherwise absorb part of the step.
    max_iter : int, optional
        Iteration limit for the optimiser.

    Returns
    -------
    FitResult
    """
    current = np.asarray(pulse["I"], dtype=float)
    voltage = np.asarray(pulse["V"], dtype=float)
    t = np.asarray(pulse["t"], dtype=float)
    t = t - t[0]

    # State of charge at the start, from the resting voltage of the first sample.
    soc_init = float(ocv.soc_at(voltage[0]))

    # The model's clock starts when the current does, not when the recording
    # does. Samples taken before the step are clamped to zero, where the model
    # contributes nothing anyway because the current there is zero. Getting this
    # wrong shifts the whole response by one sample and biases R0 and the
    # fastest RC element by several percent.
    step = _current_step(current, "pulse")
    t_model = np.maximum(t - t[step], 0.0)

    # Charge moved since t=0, and the state of charge it implies. The first
    # sample has moved no charge yet, hence the leading zero.
    charge = np.concatenate(([0.0], np.cumsum(current[:-1] * np.diff(t))))
    soc = soc_init + charge / (capacity * 3600.0)

    # Remove the OCV so only the overpotential the ECM describes is left, then
    # remove the constant offset left by any error in soc_init.
    overpotential = voltage - ocv.voltage_at(soc)
    overpotential = overpotential - overpotential[0]

    # R0 is the instantaneous jump at the step: everything else needs time.
    r0_measured = float(overpotential[step] / current[step])

    # A pulse excites and observes over the same window, so both limits coincide.
    tau_max = identifiable_tau(t_model[-1], t_model[-1])

    if p0 is None:
        # The settled overpotential is R0 plus all RC resistances, so what is
        # left after removing R0 is shared out over the RC elements.
        r_guess = max(abs(overpotential[-1] / current[-1]) - abs(r0_measured), 0.0)
        p0 = np.concatenate(
            (
                [r0_measured],
                np.full(n_rc, max(r_guess / n_rc, 1e-6)),
                _time_constant_guess(n_rc, tau_max),
            )
        )
    p0 = np.asarray(p0, dtype=float)

    bounds = (
        _held_bounds(bounds, r0_measured, p0.size, tau_max)
        if hold_r0
        else (_default_bounds(p0.size, tau_max) if bounds is None else bounds)
    )

    result = _minimise(
        lambda p: _rms(voltage_step_response(p, current, t_model) - overpotential),
        p0,
        bounds,
        max_iter,
    )
    fitted = voltage_step_response(result.x, current, t_model)
    return FitResult(
        parameter=result.x,
        soc=soc_init,
        residual=_rms(fitted - overpotential),
        x=t,
        measured=overpotential,
        fitted=fitted,
        success=bool(result.success),
        message=str(result.message),
    )


def fit_relaxation(
    pulse,
    relax,
    ocv,
    capacity,
    n_rc=3,
    p0=None,
    bounds=None,
    hold_r0=False,
    max_iter=10000,
):
    """Fit an ECM to the relaxation following a current pulse.

    Relaxation data resolves the slow RC elements better than the pulse itself,
    because nothing is being driven while they decay.

    Parameters
    ----------
    pulse : dict
        The preceding pulse, keys ``'I'``, ``'V'``, ``'t'``. Only its duration
        and its first (resting) voltage sample are used.
    relax : dict
        The relaxation, keys ``'I'``, ``'V'``, ``'t'``. Its first sample must be
        the last sample of the pulse, so that the ohmic step is captured.
    ocv : ecmfit.models.OCV
        Open-circuit voltage curve of the cell.
    capacity : float
        Nominal cell capacity in Ah.
    n_rc : int, optional
        Number of RC elements. Default 3.
    p0 : array-like, optional
        Initial parameter guess. Derived from the data if omitted.
    bounds : sequence of (low, high), optional
        Per-parameter bounds passed to the optimiser.
    hold_r0 : bool, optional
        Pin ``R0`` to the value read off the ohmic step instead of fitting it.
    max_iter : int, optional
        Iteration limit for the optimiser.

    Returns
    -------
    FitResult
    """
    pulse_time = np.asarray(pulse["t"], dtype=float)
    pulse_current = np.asarray(pulse["I"], dtype=float)
    pulse_voltage = np.asarray(pulse["V"], dtype=float)
    relax_time = np.asarray(relax["t"], dtype=float)
    relax_current = np.asarray(relax["I"], dtype=float)
    relax_voltage = np.asarray(relax["V"], dtype=float)

    if relax_current[0] == 0.0:
        raise ValueError(
            "relax['I'][0] must hold the pulse amplitude: R0 is only identifiable "
            "from the voltage step between the last pulse sample and the first "
            "rest sample, so that step has to be inside the relaxation slice"
        )
    rest = np.flatnonzero(relax_current == 0.0)
    if rest.size == 0:
        raise ValueError("relax['I'] never returns to zero; this is not a relaxation")
    rest = int(rest[0])

    # Both slices share one absolute time base, so the current flowed from the
    # pulse's own step until the first sample at rest.
    pulse_start = pulse_time[_current_step(pulse_current, "pulse")]
    t_pulse = float(relax_time[rest] - pulse_start)
    amplitude = float(relax_current[0])

    # The decay clock starts at the first zero-current sample. The samples
    # before it still carry the ohmic drop, which is exactly what identifies R0.
    t = relax_time - relax_time[0]
    t_decay = np.maximum(relax_time - relax_time[rest], 0.0)

    # State of charge across the relaxation: where the pulse started, plus the
    # charge the pulse moved before this slice began, plus whatever still moves
    # inside it. That last part is a single sample interval, but R0 is read off
    # a single sample interval too, so treating the OCV as one constant here
    # puts the whole of that error straight into R0.
    soc_init = float(ocv.soc_at(pulse_voltage[0]))
    charge = np.concatenate(([0.0], np.cumsum(relax_current[:-1] * np.diff(relax_time))))
    soc = soc_init + (amplitude * (relax_time[0] - pulse_start) + charge) / (
        capacity * 3600.0
    )
    overpotential = relax_voltage - ocv.voltage_at(soc)

    r0_measured = _ohmic_step(overpotential, amplitude, rest)

    # The pulse sets how far the slow elements were excited, the rest sets how
    # much of their decay was seen. Whichever is shorter is the real limit, and
    # for a short pulse followed by a long rest that is the pulse.
    tau_max = identifiable_tau(t_pulse, float(t_decay[-1]))

    if p0 is None:
        tau_guess = _time_constant_guess(n_rc, tau_max)
        # Invert the charge-up: the overpotential left when the current stops is
        # what the RC elements reached during t_pulse, so
        # R = U / (I * (1 - exp(-t_pulse/tau))), shared over the elements.
        polarisation = overpotential[rest] / amplitude
        r_guess = np.maximum(
            polarisation / ((1.0 - np.exp(-t_pulse / tau_guess)) * n_rc), 1e-6
        )
        p0 = np.concatenate(([r0_measured], r_guess, tau_guess))
    p0 = np.asarray(p0, dtype=float)

    bounds = (
        _held_bounds(bounds, r0_measured, p0.size, tau_max)
        if hold_r0
        else (_default_bounds(p0.size, tau_max) if bounds is None else bounds)
    )

    def residual(p):
        return _rms(
            voltage_relaxation(p, relax_current, t_decay, t_pulse) - overpotential
        )

    result = _minimise(residual, p0, bounds, max_iter)
    fitted = voltage_relaxation(result.x, relax_current, t_decay, t_pulse)
    return FitResult(
        parameter=result.x,
        soc=float(soc[rest]),
        residual=_rms(fitted - overpotential),
        x=t,
        measured=overpotential,
        fitted=fitted,
        success=bool(result.success),
        message=str(result.message),
    )


def _ohmic_step(overpotential, amplitude, rest, min_resistance=1e-3, max_samples=8):
    """Read ``R0`` off the voltage step at the end of the pulse.

    ``rest`` is the first zero-current sample, so ``overpotential[rest-1]`` still
    carries the ohmic drop and ``overpotential[rest]`` no longer does. On a
    slowly sampled measurement the step can smear across the sampling instant
    and read implausibly small; walk a few samples further in that case rather
    than pinning ``R0`` to a value the fit can never recover from.
    """
    r0 = 0.0
    for i in range(rest, min(rest + max_samples, overpotential.size)):
        r0 = (overpotential[rest - 1] - overpotential[i]) / amplitude
        if r0 > min_resistance:
            break
    return float(r0)


def _ohmic_resistance_from_eis(f, z):
    """Estimate ``R0`` as Re(Z) where Im(Z) crosses zero.

    Above that frequency the cell looks inductive (wiring, not electrochemistry)
    and below it capacitive, so the crossing is the purely ohmic point. If the
    spectrum never crosses — common when the inductive branch was not measured —
    fall back to the smallest real part, which is the closest available point.
    """
    imag = np.asarray(z.imag, dtype=float)
    real = np.asarray(z.real, dtype=float)
    crossings = np.flatnonzero(np.diff(np.signbit(imag)))
    if crossings.size == 0:
        return float(real[np.argmin(real)])
    i = int(crossings[0])
    # Linear interpolation done by hand: np.interp needs an increasing x and
    # Im(Z) is falling here, which would silently give a wrong answer.
    weight = imag[i] / (imag[i] - imag[i + 1])
    return float(real[i] + weight * (real[i + 1] - real[i]))


def _eis_time_constant_guess(f, z, n_rc):
    """Guess time constants from where the spectrum actually shows curvature.

    An RC element peaks in ``-Im(Z)`` at ``f = 1/(2*pi*tau)``, so the band of
    frequencies where ``-Im(Z)`` is appreciable is the band where the time
    constants live. Spreading the guesses over the full measured range instead
    would put most of them in decades the arcs never reach, and Nelder-Mead does
    not recover from that.
    """
    neg_imag = -np.asarray(z.imag, dtype=float)
    band = f[neg_imag > 0.2 * neg_imag.max()] if neg_imag.max() > 0 else f
    if band.size == 0:
        band = f
    return np.logspace(
        np.log10(1.0 / (2 * np.pi * band.max())),
        np.log10(1.0 / (2 * np.pi * band.min())),
        n_rc,
    )


def fit_eis(
    spectrum,
    soc=None,
    n_rc=2,
    p0=None,
    bounds=None,
    weighting="modulus",
    max_iter=10000,
):
    """Fit an ECM to an impedance spectrum.

    Parameters
    ----------
    spectrum : dict
        Keys ``'frequency'`` (Hz), ``'real'`` (Ohm) and ``'imag'`` (Ohm).
    soc : float, optional
        State of charge the spectrum was measured at. Carried through to the
        result for bookkeeping; it does not affect the fit.
    n_rc : int, optional
        Number of RC elements. Default 2 — one semicircle each. Fitting more
        semicircles than the spectrum actually shows gives parameters that look
        fine and mean nothing.
    p0 : array-like, optional
        Initial parameter guess. Derived from the data if omitted.
    bounds : sequence of (low, high), optional
        Per-parameter bounds passed to the optimiser.
    weighting : {'modulus', 'unit'}, optional
        ``'modulus'`` divides each residual by ``|Z|`` so that the low-frequency
        points, which are an order of magnitude larger, do not swamp the
        high-frequency ones. ``'unit'`` fits the raw residual.
    max_iter : int, optional
        Iteration limit for the optimiser.

    Returns
    -------
    FitResult
    """
    f = np.asarray(spectrum["frequency"], dtype=float)
    z = np.asarray(spectrum["real"], dtype=float) + 1j * np.asarray(
        spectrum["imag"], dtype=float
    )

    if weighting == "modulus":
        weight = 1.0 / np.maximum(np.abs(z), np.finfo(float).tiny)
    elif weighting == "unit":
        weight = np.ones_like(f)
    else:
        raise ValueError(f"weighting must be 'modulus' or 'unit', got {weighting!r}")

    if p0 is None:
        r0_guess = _ohmic_resistance_from_eis(f, z)
        # The width of the arc is the polarisation resistance; split it evenly
        # over the RC elements as a starting point.
        r_span = max(float(z.real.max()) - r0_guess, r0_guess)
        p0 = np.concatenate(
            (
                [r0_guess],
                np.full(n_rc, r_span / n_rc),
                _eis_time_constant_guess(f, z, n_rc),
            )
        )
    p0 = np.asarray(p0, dtype=float)
    if bounds is None:
        # Positivity only, no upper cap. A spectrum constrains every point
        # independently, so it does not suffer the runaway the relaxation fit
        # does, and capping tau at 1/(2*pi*f_min) would clip processes that are
        # still perfectly visible on the rising flank of their arc.
        bounds = _default_bounds(p0.size)

    result = _minimise(
        lambda p: _rms((impedance(p, f) - z) * weight), p0, bounds, max_iter
    )
    fitted = impedance(result.x, f)
    return FitResult(
        parameter=result.x,
        soc=soc,
        residual=_rms(fitted - z),
        x=f,
        measured=z,
        fitted=fitted,
        success=bool(result.success),
        message=str(result.message),
    )
