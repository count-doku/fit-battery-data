"""Regenerate ``docs/fits.png``, the figure shown in the README.

Committed so the figure can be checked rather than trusted::

    uv run python docs/make_figure.py
"""

import matplotlib.pyplot as plt
import numpy as np

from ecmfit import (
    OCV,
    ECMCell,
    fit_eis,
    fit_pulse,
    fit_relaxation,
    impedance,
    plot_nyquist,
    plot_time_domain,
)

PARAMETER = [0.020, 0.010, 0.030, 10.0, 180.0]  # R0, R1, R2, tau1, tau2
CAPACITY = 1.0  # Ah


def main():
    ocv = OCV([0.0, 1.0], [2.5, 4.2])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))

    # A pulse with realistic measurement noise on the voltage.
    cell = ECMCell(PARAMETER, ocv, CAPACITY, soc_init=0.5, ts=1.0)
    current = np.array([0.0] + [-1.0] * 600)
    t, voltage, _ = cell.simulate(current)
    voltage = voltage + np.random.default_rng(0).normal(0.0, 1e-3, voltage.size)
    pulse_fit = fit_pulse({"I": current, "V": voltage, "t": t}, ocv, CAPACITY, n_rc=2)
    plot_time_domain(pulse_fit, ax=axes[0], title="pulse, 1 mV noise")
    axes[0].lines[0].set(linewidth=0.7, alpha=0.65)

    # The relaxation that follows the same pulse.
    cell = ECMCell(PARAMETER, ocv, CAPACITY, soc_init=0.5, ts=1.0)
    current = np.array([0.0] + [-1.0] * 600 + [0.0] * 3000)
    t, voltage, _ = cell.simulate(current)
    data = {"I": current, "V": voltage, "t": t}
    relax_fit = fit_relaxation(
        {k: v[:601] for k, v in data.items()},
        {k: v[600:] for k, v in data.items()},
        ocv,
        CAPACITY,
        n_rc=2,
    )
    plot_time_domain(relax_fit, ax=axes[1], title="relaxation after the pulse")
    axes[1].set_xlim(0, 1200)  # the tail past this is flat and tells you nothing

    # A spectrum carrying a wiring inductance the ECM has no branch for.
    f = np.logspace(-3, 3, 100)
    z = impedance([0.009, 0.010, 0.030, 10.0, 100.0], f) + 2j * np.pi * f * 1e-6
    eis_fit = fit_eis({"frequency": f, "real": z.real, "imag": z.imag}, soc=0.5, n_rc=2)
    plot_nyquist(eis_fit, ax=axes[2], title="spectrum with wiring inductance")

    fig.tight_layout()
    fig.savefig("docs/fits.png", dpi=140, bbox_inches="tight")
    print("wrote docs/fits.png")


if __name__ == "__main__":
    main()
