"""Round-trip checks: simulate a cell with known parameters, then fit them back.

These are the only tests that matter here. A fitting routine can be wrong in
ways that still produce a plausible-looking curve — a mis-ordered parameter
vector, a sign error on the current, an initial guess that traps the optimiser
in a local minimum. Recovering the parameters that generated the data catches
all three.
"""

import numpy as np
import pytest

from ecmfit import OCV, ECMCell, fit_eis, fit_pulse, fit_relaxation, impedance

CAPACITY = 1.0  # Ah
OCV_CURVE = OCV([0.0, 1.0], [2.5, 4.2])
TRUE_PARAMETER = [0.020, 0.010, 0.030, 10.0, 180.0]  # R0, R1, R2, tau1, tau2


def _pulse(amplitude=-1.0, duration=600, rest=0, ts=1.0, soc_init=0.5):
    """Simulate one resting sample, then a current pulse, then an optional rest."""
    cell = ECMCell(TRUE_PARAMETER, OCV_CURVE, CAPACITY, soc_init=soc_init, ts=ts)
    current = np.array([0.0] + [amplitude] * duration + [0.0] * rest)
    t, voltage, _ = cell.simulate(current)
    return {"I": current, "V": voltage, "t": t}


def _relative_error(fitted, expected):
    return np.abs(np.asarray(fitted) - np.asarray(expected)) / np.abs(expected)


def test_pulse_recovers_parameters():
    result = fit_pulse(_pulse(), OCV_CURVE, CAPACITY, n_rc=2)

    assert result.success
    assert result.residual < 1e-12
    assert result.soc == pytest.approx(0.5, abs=1e-9)
    # Synthetic data has no noise and the model is the one that generated it,
    # so anything short of machine precision means a conditioning bug.
    assert np.all(_relative_error(result.parameter, TRUE_PARAMETER) < 1e-9)


def test_relaxation_recovers_parameters():
    data = _pulse(duration=600, rest=3000)
    # One resting sample, then 600 pulse samples: index 600 is the last sample
    # still carrying current, and the relaxation has to start there so that the
    # ohmic step is inside it.
    split = 600

    pulse = {k: v[: split + 1] for k, v in data.items()}
    relax = {k: v[split:] for k, v in data.items()}
    result = fit_relaxation(pulse, relax, OCV_CURVE, CAPACITY, n_rc=2)

    assert result.success
    assert result.residual < 1e-12
    assert result.soc == pytest.approx(0.5 - 600 / 3600, abs=1e-9)

    # R0 is identified from the single sample interval in which the current
    # stops. Within that one interval the zero-order-hold simulator and the
    # continuous model cannot agree exactly, which puts a floor of a few parts
    # in 10^4 under R0 and nothing at all under the rest.
    error = _relative_error(result.parameter, TRUE_PARAMETER)
    assert error[0] < 1e-3
    assert np.all(error[1:] < 1e-9)


def test_eis_recovers_parameters():
    f = np.logspace(-3, 3, 120)
    z = impedance(TRUE_PARAMETER, f)
    spectrum = {"frequency": f, "real": z.real, "imag": z.imag}

    result = fit_eis(spectrum, soc=0.5, n_rc=2)

    assert result.success
    assert result.residual < 1e-12
    assert np.all(_relative_error(result.parameter, TRUE_PARAMETER) < 1e-9)


def test_pulse_and_eis_agree_on_the_same_cell():
    """The two measurement domains must identify the same cell."""
    f = np.logspace(-3, 3, 120)
    z = impedance(TRUE_PARAMETER, f)

    from_pulse = fit_pulse(_pulse(), OCV_CURVE, CAPACITY, n_rc=2).parameter
    from_eis = fit_eis({"frequency": f, "real": z.real, "imag": z.imag}, n_rc=2).parameter

    assert np.all(_relative_error(from_pulse, from_eis) < 1e-6)


def test_hold_r0_pins_the_series_resistance():
    result = fit_pulse(_pulse(), OCV_CURVE, CAPACITY, n_rc=2, hold_r0=True)

    # R0 is read off the ohmic step rather than fitted, so it must come back
    # exactly as measured, not merely close.
    assert result.parameter[0] == pytest.approx(TRUE_PARAMETER[0], rel=1e-9)


def test_eis_falls_back_when_the_spectrum_has_no_zero_crossing():
    """Without an inductive branch there is no Im(Z) = 0 point to interpolate."""
    f = np.logspace(-2, 2, 80)
    z = impedance(TRUE_PARAMETER, f)  # purely capacitive, Im(Z) < 0 throughout
    assert np.all(z.imag < 0)

    result = fit_eis({"frequency": f, "real": z.real, "imag": z.imag}, n_rc=2)

    assert np.all(_relative_error(result.parameter, TRUE_PARAMETER) < 1e-9)


def test_relaxation_rejects_a_slice_that_starts_after_the_step():
    """Without the current step in the slice, R0 is not identifiable at all."""
    data = _pulse(duration=600, rest=3000)
    pulse = {k: v[:601] for k, v in data.items()}
    relax = {k: v[601:] for k, v in data.items()}  # starts at rest, step missed

    with pytest.raises(ValueError, match="pulse amplitude"):
        fit_relaxation(pulse, relax, OCV_CURVE, CAPACITY, n_rc=2)


def test_ocv_curve_inverts():
    assert OCV_CURVE.soc_at(OCV_CURVE.voltage_at(0.37)) == pytest.approx(0.37)


def test_ocv_rejects_a_non_invertible_curve():
    with pytest.raises(ValueError, match="strictly increasing"):
        OCV([0.0, 0.5, 1.0], [2.5, 4.2, 4.0])


def test_pulse_survives_measurement_noise():
    """1 mV of noise is what a decent cell-voltage measurement actually has.

    Noise is where a fit stops being arithmetic and starts being an estimate:
    the residual can no longer go to zero, and the fast time constant is the
    first parameter to suffer because the samples that resolve it are few.
    """
    rng = np.random.default_rng(0)
    data = _pulse()
    data["V"] = data["V"] + rng.normal(0.0, 1e-3, data["V"].size)

    result = fit_pulse(data, OCV_CURVE, CAPACITY, n_rc=2)

    assert result.success
    # Loose on purpose. With this much noise the fast time constant lands
    # around 20 % out and no amount of optimiser tuning changes that — the
    # information is not in the data. A broken fit misses by orders of
    # magnitude, which is what this still catches.
    assert np.all(_relative_error(result.parameter, TRUE_PARAMETER) < 0.25)
    # The residual has to settle at the noise floor: much above means the fit
    # failed, much below means it is fitting the noise.
    assert 0.5e-3 < result.residual < 2e-3
