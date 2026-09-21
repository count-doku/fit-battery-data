"""Fit a 2nd order ECM to the relaxation that follows a current pulse.

Relaxation resolves slow time constants better than the pulse itself, because
nothing is being driven while the RC elements decay. Run with::

    uv run python examples/fit_relaxation.py
"""

import matplotlib.pyplot as plt
import numpy as np

from ecmfit import OCV, ECMCell, fit_relaxation, plot_time_domain

TRUE_PARAMETER = [0.020, 0.010, 0.030, 10.0, 180.0]  # R0, R1, R2, tau1, tau2
CAPACITY = 1.0  # Ah


def main():
    ocv = OCV([0.0, 1.0], [2.5, 4.2])
    cell = ECMCell(TRUE_PARAMETER, ocv, CAPACITY, soc_init=0.5, ts=1.0)

    # Rest, 10 minute discharge, then 50 minutes of relaxation.
    current = np.array([0.0] + [-1.0] * 600 + [0.0] * 3000)
    t, voltage, _ = cell.simulate(current)
    data = {"I": current, "V": voltage, "t": t}

    # Index 600 is the last sample still carrying current. The relaxation slice
    # has to start there, because the voltage step between that sample and the
    # next one is the only place R0 is visible.
    split = 600
    pulse = {k: v[: split + 1] for k, v in data.items()}
    relax = {k: v[split:] for k, v in data.items()}

    result = fit_relaxation(pulse, relax, ocv, CAPACITY, n_rc=2)

    print(f"converged: {result.success} ({result.message})")
    print(f"residual:  {result.residual:.3e} V")
    print(f"SOC:       {result.soc:.4f}\n")
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
