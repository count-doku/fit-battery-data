"""Round-trip checks: simulate a cell with known parameters, then fit them back.

These are the only tests that matter here. A fitting routine can be wrong in
ways that still produce a plausible-looking curve — a mis-ordered parameter
vector, a sign error on the current, an initial guess that traps the optimiser
in a local minimum. Recovering the parameters that generated the data catches
all three.
"""

import numpy as np
import pytest

from ecmfit import (
    OCV,
    ECMCell,
    fit_eis,
    fit_pulse,
    fit_relaxation,
    identifiable_tau,
    impedance,
)

CAPACITY = 1.0  # Ah
OCV_CURVE = OCV([0.0, 1.0], [2.5, 4.2])
TRUE_PARAMETER = [0.020, 0.010, 0.030, 10.0, 180.0]  # R0, R1, R2, tau1, tau2


def _pulse(amplitude=-1.0, duration=600, rest=0, ts=1.0, soc_init=0.5):
    """Simulate one resting sample, then a current pulse, then an optional rest."""
    cell = ECMCell(TRUE_PARAMETER, OCV_CURVE, CAPACITY, soc_init=soc_init, ts=ts)
    current = np.array([0.0] + [amplitude] * duration + [0.0] * rest)
    t, voltage, _ = cell.simulate(current)
    return {"I": current, "V": voltage, "t": t}


def _with_noise(data, sigma=1e-3, seed=0):
    """Add Gaussian noise to the voltage only.

    The current is left exact. A cycler commands current and measures voltage,
    so voltage carries the dominant error, and least squares assumes the error
    sits in the dependent variable. Note what this leaves out: a real cycler at
    commanded zero still passes a small offset current, which drifts the voltage
    over a long rest in the same time band as the slow RC element. That error
    source is not modelled here.
    """
    noisy = dict(data)
    noisy["V"] = data["V"] + np.random.default_rng(seed).normal(
        0.0, sigma, data["V"].size
    )
    return noisy


def _split_at_step(data, split=600):
    """Split one trace into the pulse and the relaxation that follows it.

    One trace, so the two slices carry the same noise realisation — which is
    the only physically possible arrangement, since a relaxation is the same
    instrument continuing to record the same cell.
    """
    return (
        {k: v[: split + 1] for k, v in data.items()},
        {k: v[split:] for k, v in data.items()},
    )


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
    # One resting sample, then 600 pulse samples: index 600 is the last sample
    # still carrying current, and the relaxation has to start there so that the
    # ohmic step is inside it.
    pulse, relax = _split_at_step(_pulse(duration=600, rest=3000))
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
    result = fit_pulse(_with_noise(_pulse()), OCV_CURVE, CAPACITY, n_rc=2)

    assert result.success
    # Loose on purpose. With this much noise the fast time constant lands
    # around 20 % out and no amount of optimiser tuning changes that — the
    # information is not in the data. A broken fit misses by orders of
    # magnitude, which is what this still catches.
    assert np.all(_relative_error(result.parameter, TRUE_PARAMETER) < 0.25)
    # The residual has to settle at the noise floor: much above means the fit
    # failed, much below means it is fitting the noise.
    assert 0.5e-3 < result.residual < 2e-3


def test_relaxation_survives_measurement_noise():
    """The same noise the pulse test uses, on the same trace, after the step.

    A relaxation is harder than a pulse and this pins down how much harder.
    Two structural reasons: R0 is carried by the single sample interval in
    which the current stops, so its noise lands at full strength with nothing
    to average against; and the signal decays towards zero, so the later
    samples are mostly noise. Reported as a median over seeds because a
    minority of fits diverge outright — see the next test for why.
    """
    errors = []
    for seed in range(12):
        pulse, relax = _split_at_step(
            _with_noise(_pulse(duration=600, rest=3000), seed=seed)
        )
        result = fit_relaxation(pulse, relax, OCV_CURVE, CAPACITY, n_rc=2)

        # Whatever the parameters do, the fit must reach the noise floor.
        assert result.success
        assert 0.5e-3 < result.residual < 2e-3
        errors.append(_relative_error(result.parameter, TRUE_PARAMETER))

    errors = np.array(errors)
    median = np.median(errors, axis=0)

    # A typical fit gets the resistances to ~10 % and tau2 to ~12 %. tau1 is
    # worse: at ts = 1 s only a handful of samples resolve a 10 s decay.
    assert np.all(median[:3] < 0.15), f"resistances: {median[:3]}"
    assert median[4] < 0.20, f"tau2: {median[4]}"
    assert median[3] < 0.50, f"tau1: {median[3]}"

    # A minority run away entirely. This is a property of separating a sum of
    # exponentials under noise, not a bug to be tuned out, but it must not get
    # worse — without the tau ceiling from identifiable_tau it was 40 %.
    diverged = (errors.max(axis=1) > 0.5).mean()
    assert diverged <= 0.35, f"diverged fraction: {diverged:.0%}"


def test_slow_element_needs_a_pulse_long_enough_to_excite_it():
    """A long rest cannot rescue a time constant the pulse never stirred.

    An RC element only charges to ``1 - exp(-t_pulse/tau)``. With tau2 = 180 s
    a 60 s pulse reaches 28 % of full amplitude, which noise then swamps; a
    600 s pulse reaches 96 %. The relaxation is 3000 s in both cases, so the
    observation window is identical and only the excitation differs.
    """

    def median_tau2_error(t_pulse):
        errors = []
        for seed in range(8):
            data = _with_noise(_pulse(duration=t_pulse, rest=3000), seed=seed)
            pulse, relax = _split_at_step(data, split=t_pulse)
            result = fit_relaxation(pulse, relax, OCV_CURVE, CAPACITY, n_rc=2)
            errors.append(_relative_error(result.parameter, TRUE_PARAMETER)[4])
        return float(np.median(errors))

    barely_excited = median_tau2_error(60)  # 28 % of full amplitude
    well_excited = median_tau2_error(600)  # 96 %

    assert well_excited < 0.25
    assert barely_excited > 4 * well_excited, (
        f"under-excited {barely_excited:.1%} vs excited {well_excited:.1%}"
    )


def test_identifiable_tau_takes_the_shorter_of_the_two_windows():
    assert identifiable_tau(t_pulse=60.0, t_observed=3000.0) == 60.0
    assert identifiable_tau(t_pulse=3000.0, t_observed=600.0) == 600.0
