"""Fit a 2nd order ECM to a current pulse.

Generates a pulse from known parameters, fits it back, and prints how well the
parameters were recovered. Run with::

    uv run python examples/fit_pulse.py
"""

import matplotlib.pyplot as plt
import numpy as np

from ecmfit import OCV, ECMCell, fit_pulse, plot_time_domain

TRUE_PARAMETER = [0.020, 0.010, 0.030, 10.0, 180.0]  # R0, R1, R2, tau1, tau2
CAPACITY = 1.0  # Ah


def main():
    ocv = OCV([0.0, 1.0], [2.5, 4.2])
    cell = ECMCell(TRUE_PARAMETER, ocv, CAPACITY, soc_init=0.5, ts=1.0)

    # One sample at rest, then a 10 minute 1C discharge. The resting sample is
    # what fixes the starting state of charge.
    current = np.array([0.0] + [-1.0] * 600)
    t, voltage, _ = cell.simulate(current)

    result = fit_pulse({"I": current, "V": voltage, "t": t}, ocv, CAPACITY, n_rc=2)

    print(f"converged: {result.success} ({result.message})")
    print(f"residual:  {result.residual:.3e} V\n")
    print(f"{'':>6}{'true':>12}{'fitted':>12}")
    for name, true, fitted in zip(
        ["R0", "R1", "R2", "tau1", "tau2"],
        TRUE_PARAMETER,
        result.parameter,
        strict=True,
    ):
        print(f"{name:>6}{true:>12.4g}{fitted:>12.4g}")

    plot_time_domain(result)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
