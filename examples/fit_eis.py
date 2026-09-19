"""Fit a 2nd order ECM to an impedance spectrum.

The spectrum here also carries a series inductance, which real cells show above
roughly 1 kHz. The ECM has no inductive branch, so the fit is deliberately
imperfect at the top of the frequency range — that mismatch is what the
inductive tail looks like in practice, and it is why R0 is taken from the
zero crossing of Im(Z) rather than from the highest measured frequency.

Run with::

    uv run python examples/fit_eis.py
"""

import matplotlib.pyplot as plt
import numpy as np

from ecmfit import fit_eis, impedance, plot_nyquist

TRUE_PARAMETER = [0.009, 0.010, 0.030, 10.0, 100.0]  # R0, R1, R2, tau1, tau2
INDUCTANCE = 1e-6  # H, wiring and cell geometry


def main():
    f = np.logspace(-3, 3, 100)
    z = impedance(TRUE_PARAMETER, f) + 2j * np.pi * f * INDUCTANCE

    result = fit_eis({"frequency": f, "real": z.real, "imag": z.imag}, soc=0.5, n_rc=2)

    print(f"converged: {result.success} ({result.message})")
    print(f"residual:  {result.residual:.3e} Ohm\n")
    print(f"{'':>6}{'true':>12}{'fitted':>12}")
    for name, true, fitted in zip(
        ["R0", "R1", "R2", "tau1", "tau2"],
        TRUE_PARAMETER,
        result.parameter,
        strict=True,
    ):
        print(f"{name:>6}{true:>12.4g}{fitted:>12.4g}")

    plot_nyquist(result)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
