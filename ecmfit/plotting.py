"""Plots for inspecting a fit. Kept out of the fitting code on purpose.

A fitting function that draws its own figures cannot be used in a loop over
several hundred pulses without either opening several hundred windows or
growing a pile of plotting keyword arguments. The fit returns its data; these
helpers draw it.
"""

import matplotlib.pyplot as plt

__all__ = ["plot_nyquist", "plot_time_domain"]


def plot_time_domain(result, ax=None, title=None):
    """Plot measured and fitted overpotential over time.

    Parameters
    ----------
    result : ecmfit.fit.FitResult
        Result of :func:`~ecmfit.fit.fit_pulse` or
        :func:`~ecmfit.fit.fit_relaxation`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    title : str, optional
        Overrides the default title, which reports SOC and residual.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(result.x, result.measured, label="measured")
    ax.plot(result.x, result.fitted, "--", label="fit")
    ax.set(
        xlabel="time in s",
        ylabel="overpotential in V",
        title=title or _default_title(result, "V"),
    )
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def plot_nyquist(result, ax=None, title=None):
    """Plot measured and fitted impedance as a Nyquist diagram.

    Parameters
    ----------
    result : ecmfit.fit.FitResult
        Result of :func:`~ecmfit.fit.fit_eis`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if omitted.
    title : str, optional
        Overrides the default title, which reports SOC and residual.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(result.measured.real, -result.measured.imag, label="measured")
    ax.plot(result.fitted.real, -result.fitted.imag, "--", label="fit")
    ax.set(
        xlabel=r"Re(Z) in $\Omega$",
        ylabel=r"$-$Im(Z) in $\Omega$",
        title=title or _default_title(result, r"$\Omega$"),
    )
    # Equal aspect: a semicircle has to look like one, otherwise the plot is
    # useless for judging the fit by eye.
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def _default_title(result, unit):
    soc = "" if result.soc is None else f"SOC {result.soc:.2f}, "
    return f"{soc}residual {result.residual:.2e} {unit}"
