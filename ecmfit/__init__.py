"""Fit equivalent-circuit model parameters to lithium-ion battery measurements."""

from ecmfit.fit import (
    FitResult,
    fit_eis,
    fit_pulse,
    fit_relaxation,
    identifiable_tau,
)
from ecmfit.models import OCV, impedance, voltage_relaxation, voltage_step_response
from ecmfit.plotting import plot_nyquist, plot_time_domain
from ecmfit.simulate import ECMCell

__version__ = "0.2.0"

__all__ = [
    "ECMCell",
    "FitResult",
    "OCV",
    "fit_eis",
    "fit_pulse",
    "fit_relaxation",
    "identifiable_tau",
    "impedance",
    "plot_nyquist",
    "plot_time_domain",
    "voltage_relaxation",
    "voltage_step_response",
]
